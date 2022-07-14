use crate::component::func::{self, Lift, Lower, Memory, MemoryMut, Options, WasmStr};
use crate::store::StoreOpaque;
use crate::{AsContextMut, StoreContextMut, ValRaw};
use anyhow::{anyhow, bail, Result};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;
use wasmtime_component_util::{DiscriminantSize, FlagsSize};
use wasmtime_environ::component::{ComponentTypes, InterfaceType};

/// Represents an owned, despecialized version of `InterfaceType`
///
/// This type serves two purposes:
///
/// - It allows us to despecialize as the first step, which reduces the number of cases to consider when
/// typechecking and lowering.
///
/// - It avoids needing to borrow the store both mutably and immutably when lowering, as we would if we used
/// `InterfaceType`s.
#[derive(Debug)]
pub(crate) enum Type {
    Bool,
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    S64,
    U64,
    Float32,
    Float64,
    Char,
    String,
    List(Box<Type>),
    Record(Box<[Type]>),
    Variant(Box<[Type]>),
    Flags(usize),
}

impl Type {
    /// Convert the specified `InterfaceType` to a `Type`.
    pub(crate) fn from(ty: &InterfaceType, types: &ComponentTypes) -> Self {
        match ty {
            InterfaceType::Unit => Type::Record(Box::new([])),
            InterfaceType::Bool => Type::Bool,
            InterfaceType::S8 => Type::S8,
            InterfaceType::U8 => Type::U8,
            InterfaceType::S16 => Type::S16,
            InterfaceType::U16 => Type::U16,
            InterfaceType::S32 => Type::S32,
            InterfaceType::U32 => Type::U32,
            InterfaceType::S64 => Type::S64,
            InterfaceType::U64 => Type::U64,
            InterfaceType::Float32 => Type::Float32,
            InterfaceType::Float64 => Type::Float64,
            InterfaceType::Char => Type::Char,
            InterfaceType::String => Type::String,
            InterfaceType::Record(index) => Type::Record(
                types[*index]
                    .fields
                    .iter()
                    .map(|field| Type::from(&field.ty, types))
                    .collect(),
            ),
            InterfaceType::Variant(index) => Type::Variant(
                types[*index]
                    .cases
                    .iter()
                    .map(|case| Type::from(&case.ty, types))
                    .collect(),
            ),
            InterfaceType::List(index) => Type::List(Box::new(Type::from(&types[*index], types))),
            InterfaceType::Tuple(index) => Type::Record(
                types[*index]
                    .types
                    .iter()
                    .map(|ty| Type::from(ty, types))
                    .collect(),
            ),
            InterfaceType::Flags(index) => Type::Flags(types[*index].names.len()),
            InterfaceType::Enum(index) => Type::Variant(
                types[*index]
                    .names
                    .iter()
                    .map(|_| Type::Record(Box::new([])))
                    .collect(),
            ),
            InterfaceType::Union(index) => Type::Variant(
                types[*index]
                    .types
                    .iter()
                    .map(|ty| Type::from(ty, types))
                    .collect(),
            ),
            InterfaceType::Option(index) => Type::Variant(Box::new([
                Type::Record(Box::new([])),
                Type::from(&types[*index], types),
            ])),
            InterfaceType::Expected(index) => {
                let expected = &types[*index];

                Type::Variant(Box::new([
                    Type::from(&expected.ok, types),
                    Type::from(&expected.err, types),
                ]))
            }
        }
    }

    /// Return the number of stack slots needed to store values of this type in lowered form.
    pub(crate) fn flatten_count(&self) -> usize {
        match self {
            Type::Bool
            | Type::S8
            | Type::U8
            | Type::S16
            | Type::U16
            | Type::S32
            | Type::U32
            | Type::S64
            | Type::U64
            | Type::Float32
            | Type::Float64
            | Type::Char => 1,

            Type::String | Type::List(_) => 2,

            Type::Record(types) => types.iter().map(Type::flatten_count).sum(),

            Type::Variant(types) => 1 + types.iter().map(Type::flatten_count).max().unwrap_or(0),

            Type::Flags(count) => u32_count_for_flag_count(*count),
        }
    }

    fn desc(&self) -> &'static str {
        match self {
            Type::Bool => "bool",
            Type::S8 => "s8",
            Type::U8 => "u8",
            Type::S16 => "s16",
            Type::U16 => "u16",
            Type::S32 => "s32",
            Type::U32 => "u32",
            Type::S64 => "s64",
            Type::U64 => "u64",
            Type::Float32 => "float32",
            Type::Float64 => "float64",
            Type::Char => "char",
            Type::String => "string",
            Type::List(_) => "list",
            Type::Record(_) => "record",
            Type::Variant(_) => "variant",
            Type::Flags(_) => "flags",
        }
    }
}

/// Possible runtime values which a component function can either consume or produce
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Val {
    /// Boolean
    Bool(bool),
    /// Signed 8-bit integer
    S8(i8),
    /// Unsigned 8-bit integer
    U8(u8),
    /// Signed 16-bit integer
    S16(i16),
    /// Unsigned 16-bit integer
    U16(u16),
    /// Signed 32-bit integer
    S32(i32),
    /// Unsigned 32-bit integer
    U32(u32),
    /// Signed 64-bit integer
    S64(i64),
    /// Unsigned 64-bit integer
    U64(u64),
    /// 32-bit floating point value
    Float32(u32),
    /// 64-bit floating point value
    Float64(u64),
    /// 32-bit character
    Char(char),
    /// Character string
    String(Rc<str>),
    /// List of values
    List(Rc<[Val]>),
    /// Record, tuple, or unit
    Record(Rc<[Val]>),
    /// Variant, enum, or union
    Variant {
        /// Index of case
        discriminant: u32,
        /// Associated data for case
        value: Rc<Val>,
    },
    /// Bit flags
    Flags {
        /// Total number of flags
        count: u32,
        /// Values of flags
        value: Rc<[u32]>,
    },
}

impl Val {
    /// Verify that this value fits the specified type.
    pub(crate) fn typecheck(&self, ty: &Type) -> Result<()> {
        match (self, ty) {
            (Val::Bool(_), Type::Bool)
            | (Val::S8(_), Type::S8)
            | (Val::U8(_), Type::U8)
            | (Val::S16(_), Type::S16)
            | (Val::U16(_), Type::U16)
            | (Val::S32(_), Type::S32)
            | (Val::U32(_), Type::U32)
            | (Val::S64(_), Type::S64)
            | (Val::U64(_), Type::U64)
            | (Val::Float32(_), Type::Float32)
            | (Val::Float64(_), Type::Float64)
            | (Val::Char(_), Type::Char)
            | (Val::String(_), Type::String) => (),

            (Val::List(items), Type::List(element_type)) => {
                for item in items.deref() {
                    item.typecheck(element_type)?;
                }
            }

            (Val::Record(values), Type::Record(types)) => {
                if types.len() != values.len() {
                    bail!(
                        "expected {} field values, got {}",
                        types.len(),
                        values.len()
                    );
                }

                for (ty, value) in types.iter().zip(values.deref()) {
                    value.typecheck(ty)?;
                }
            }

            (
                Val::Variant {
                    discriminant,
                    value,
                },
                Type::Variant(types),
            ) => {
                if *discriminant as usize >= types.len() {
                    bail!(
                        "discriminant {} is out of expected range [0, {})",
                        discriminant,
                        types.len()
                    );
                }

                value.typecheck(&types[*discriminant as usize])?;
            }

            (Val::Flags { count, value }, Type::Flags(type_count)) => {
                let type_count = *type_count;
                let count = *count as usize;

                if count > value.len() * 32 {
                    bail!(
                        "flag count {} must not be larger than value count {}",
                        count,
                        value.len() * 32
                    );
                }

                if u32_count_for_flag_count(count) > value.len() {
                    bail!(
                        "value count {} must not be larger than required by flag count {}",
                        value.len() * 32,
                        count,
                    );
                }

                if type_count != count {
                    bail!("expected {} flags, got {}", type_count, count);
                }
            }

            _ => bail!("type mismatch: expected {}, got {}", ty.desc(), self.desc()),
        }

        Ok(())
    }

    fn desc(&self) -> &'static str {
        match self {
            Val::Bool(_) => "bool",
            Val::S8(_) => "s8",
            Val::U8(_) => "u8",
            Val::S16(_) => "s16",
            Val::U16(_) => "u16",
            Val::S32(_) => "s32",
            Val::U32(_) => "u32",
            Val::S64(_) => "s64",
            Val::U64(_) => "u64",
            Val::Float32(_) => "float32",
            Val::Float64(_) => "float64",
            Val::Char(_) => "char",
            Val::String(_) => "string",
            Val::List(_) => "list",
            Val::Record(_) => "record",
            Val::Variant { .. } => "variant",
            Val::Flags { .. } => "flags",
        }
    }

    /// Serialized this value as core Wasm stack values.
    pub(crate) fn lower<T>(
        &self,
        store: &mut StoreContextMut<T>,
        options: &Options,
        ty: &Type,
        vec: &mut Vec<ValRaw>,
    ) -> Result<()> {
        match (self, ty) {
            (Val::Bool(value), Type::Bool) => vec.push(ValRaw::u32(if *value { 1 } else { 0 })),
            (Val::S8(value), Type::S8) => vec.push(ValRaw::i32(*value as i32)),
            (Val::U8(value), Type::U8) => vec.push(ValRaw::u32(*value as u32)),
            (Val::S16(value), Type::S16) => vec.push(ValRaw::i32(*value as i32)),
            (Val::U16(value), Type::U16) => vec.push(ValRaw::u32(*value as u32)),
            (Val::S32(value), Type::S32) => vec.push(ValRaw::i32(*value)),
            (Val::U32(value), Type::U32) => vec.push(ValRaw::u32(*value)),
            (Val::S64(value), Type::S64) => vec.push(ValRaw::i64(*value)),
            (Val::U64(value), Type::U64) => vec.push(ValRaw::u64(*value)),
            (Val::Float32(value), Type::Float32) => vec.push(ValRaw::f32(*value)),
            (Val::Float64(value), Type::Float64) => vec.push(ValRaw::f64(*value)),
            (Val::Char(value), Type::Char) => vec.push(ValRaw::u32(u32::from(*value))),
            (Val::String(value), Type::String) => {
                let (ptr, len) = super::lower_string(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    value,
                )?;
                vec.push(ValRaw::i64(ptr as i64));
                vec.push(ValRaw::i64(len as i64));
            }
            (Val::List(items), Type::List(element_type)) => {
                let (ptr, len) = lower_list(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    element_type,
                    &items,
                )?;
                vec.push(ValRaw::i64(ptr as i64));
                vec.push(ValRaw::i64(len as i64));
            }
            (Val::Record(values), Type::Record(types)) => {
                for (value, ty) in values.iter().zip(types.deref()) {
                    value.lower(store, options, ty, vec)?;
                }
            }
            (
                Val::Variant {
                    discriminant,
                    value,
                },
                Type::Variant(types),
            ) => {
                vec.push(ValRaw::u32(*discriminant));
                value.lower(store, options, &types[*discriminant as usize], vec)?;
                // Pad `vec` out to max payload size:
                vec.extend(
                    iter::repeat(ValRaw::u32(0))
                        .take(ty.flatten_count().checked_sub(vec.len()).unwrap()),
                );
            }
            (Val::Flags { value, .. }, Type::Flags(_)) => {
                vec.extend(value.iter().map(|&v| ValRaw::u32(v)));
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Serialize this value to the heap at the specified memory location.
    pub(crate) fn store<T>(
        &self,
        mem: &mut MemoryMut<'_, T>,
        ty: &Type,
        offset: usize,
    ) -> Result<()> {
        match (self, ty) {
            (Val::Bool(value), Type::Bool) => value.store(mem, offset)?,
            (Val::S8(value), Type::S8) => value.store(mem, offset)?,
            (Val::U8(value), Type::U8) => value.store(mem, offset)?,
            (Val::S16(value), Type::S16) => value.store(mem, offset)?,
            (Val::U16(value), Type::U16) => value.store(mem, offset)?,
            (Val::S32(value), Type::S32) => value.store(mem, offset)?,
            (Val::U32(value), Type::U32) => value.store(mem, offset)?,
            (Val::S64(value), Type::S64) => value.store(mem, offset)?,
            (Val::U64(value), Type::U64) => value.store(mem, offset)?,
            (Val::Float32(value), Type::Float32) => value.store(mem, offset)?,
            (Val::Float64(value), Type::Float64) => value.store(mem, offset)?,
            (Val::Char(value), Type::Char) => value.store(mem, offset)?,
            (Val::String(value), Type::String) => {
                let (ptr, len) = super::lower_string(mem, value)?;
                // FIXME: needs memory64 handling
                *mem.get(offset + 0) = (ptr as i32).to_le_bytes();
                *mem.get(offset + 4) = (len as i32).to_le_bytes();
            }
            (Val::List(items), Type::List(element_type)) => {
                let (ptr, len) = lower_list(mem, element_type, &items)?;
                // FIXME: needs memory64 handling
                *mem.get(offset + 0) = (ptr as i32).to_le_bytes();
                *mem.get(offset + 4) = (len as i32).to_le_bytes();
            }
            (Val::Record(values), Type::Record(types)) => {
                let mut offset = offset;
                for (value, ty) in values.iter().zip(types.deref()) {
                    value.store(mem, ty, next_field(ty, &mut offset))?;
                }
            }
            (
                Val::Variant {
                    discriminant,
                    value,
                },
                Type::Variant(types),
            ) => {
                let case_ty = &types[*discriminant as usize];
                let discriminant_size = DiscriminantSize::from_count(types.len()).unwrap();
                match discriminant_size {
                    DiscriminantSize::Size1 => {
                        u8::try_from(*discriminant).unwrap().store(mem, offset)?
                    }
                    DiscriminantSize::Size2 => {
                        u16::try_from(*discriminant).unwrap().store(mem, offset)?
                    }
                    DiscriminantSize::Size4 => (*discriminant).store(mem, offset)?,
                }

                value.store(
                    mem,
                    case_ty,
                    offset
                        + func::align_to(
                            discriminant_size.into(),
                            SizeAndAlignment::from(ty).alignment,
                        ),
                )?;
            }
            (Val::Flags { count, value }, Type::Flags(_)) => {
                match FlagsSize::from_count(*count as usize) {
                    FlagsSize::Size1 => u8::try_from(value[0]).unwrap().store(mem, offset)?,
                    FlagsSize::Size2 => u16::try_from(value[0]).unwrap().store(mem, offset)?,
                    FlagsSize::Size4Plus(_) => {
                        let mut offset = offset;
                        for value in value.deref() {
                            value.store(mem, offset)?;
                            offset += 4;
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Deserialize this value from core Wasm stack values.
    pub(crate) fn lift<'a>(
        store: &StoreOpaque,
        options: &Options,
        ty: &Type,
        src: &mut impl Iterator<Item = &'a ValRaw>,
    ) -> Result<Self> {
        fn next<'a>(src: &mut impl Iterator<Item = &'a ValRaw>) -> &'a ValRaw {
            src.next().unwrap()
        }

        Ok(match ty {
            Type::Bool => Val::Bool(bool::lift(store, options, next(src))?),
            Type::S8 => Val::S8(i8::lift(store, options, next(src))?),
            Type::U8 => Val::U8(u8::lift(store, options, next(src))?),
            Type::S16 => Val::S16(i16::lift(store, options, next(src))?),
            Type::U16 => Val::U16(u16::lift(store, options, next(src))?),
            Type::S32 => Val::S32(i32::lift(store, options, next(src))?),
            Type::U32 => Val::U32(u32::lift(store, options, next(src))?),
            Type::S64 => Val::S64(i64::lift(store, options, next(src))?),
            Type::U64 => Val::U64(u64::lift(store, options, next(src))?),
            Type::Float32 => Val::Float32(u32::lift(store, options, next(src))?),
            Type::Float64 => Val::Float64(u64::lift(store, options, next(src))?),
            Type::Char => Val::Char(char::lift(store, options, next(src))?),
            Type::String | Type::List(_) => {
                // These won't fit in func::MAX_STACK_RESULTS as of this writing, so presumably we should never
                // reach here
                unreachable!()
            }
            Type::Record(types) => Val::Record(
                types
                    .into_iter()
                    .map(|ty| Self::lift(store, options, ty, src))
                    .collect::<Result<_>>()?,
            ),
            Type::Variant(types) => {
                let discriminant = next(src).get_u32();
                let case_ty = types.get(discriminant as usize).ok_or_else(|| {
                    anyhow!(
                        "discriminant {} out of range [0..{})",
                        discriminant,
                        types.len()
                    )
                })?;
                let value = Rc::new(Self::lift(store, options, case_ty, src)?);

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            Type::Flags(count) => {
                assert!(*count <= 32);
                let value = iter::once(u32::lift(store, options, next(src))?).collect();

                Val::Flags {
                    count: u32::try_from(*count)?,
                    value,
                }
            }
        })
    }

    /// Deserialized this value from the heap.
    pub(crate) fn load(store: &StoreOpaque, mem: &Memory, ty: &Type, bytes: &[u8]) -> Result<Self> {
        Ok(match ty {
            Type::Bool => Val::Bool(bool::load(mem, bytes)?),
            Type::S8 => Val::S8(i8::load(mem, bytes)?),
            Type::U8 => Val::U8(u8::load(mem, bytes)?),
            Type::S16 => Val::S16(i16::load(mem, bytes)?),
            Type::U16 => Val::U16(u16::load(mem, bytes)?),
            Type::S32 => Val::S32(i32::load(mem, bytes)?),
            Type::U32 => Val::U32(u32::load(mem, bytes)?),
            Type::S64 => Val::S64(i64::load(mem, bytes)?),
            Type::U64 => Val::U64(u64::load(mem, bytes)?),
            Type::Float32 => Val::Float32(u32::load(mem, bytes)?),
            Type::Float64 => Val::Float64(u64::load(mem, bytes)?),
            Type::Char => Val::Char(char::load(mem, bytes)?),
            Type::String => {
                Val::String(Rc::from(WasmStr::load(mem, bytes)?._to_str(store)?.deref()))
            }
            Type::List(element_type) => {
                // FIXME: needs memory64 treatment
                let ptr = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
                let len = u32::from_le_bytes(bytes[4..].try_into().unwrap()) as usize;
                let SizeAndAlignment {
                    size: element_size,
                    alignment: element_alignment,
                } = SizeAndAlignment::from(element_type);

                match len
                    .checked_mul(element_size)
                    .and_then(|len| ptr.checked_add(len))
                {
                    Some(n) if n <= mem.as_slice().len() => {}
                    _ => bail!("list pointer/length out of bounds of memory"),
                }
                if ptr % usize::try_from(element_alignment)? != 0 {
                    bail!("list pointer is not aligned")
                }

                Val::List(
                    (0..len)
                        .map(|index| {
                            Self::load(
                                store,
                                mem,
                                element_type,
                                &mem.as_slice()[ptr + (index * element_size)..][..element_size],
                            )
                        })
                        .collect::<Result<_>>()?,
                )
            }
            Type::Record(types) => {
                let mut offset = 0;
                Val::Record(
                    types
                        .into_iter()
                        .map(|ty| {
                            Self::load(
                                store,
                                mem,
                                ty,
                                &bytes[next_field(ty, &mut offset)..]
                                    [..SizeAndAlignment::from(ty).size],
                            )
                        })
                        .collect::<Result<_>>()?,
                )
            }
            Type::Variant(types) => {
                let discriminant_size = DiscriminantSize::from_count(types.len()).unwrap();
                let discriminant = match discriminant_size {
                    DiscriminantSize::Size1 => u8::load(mem, &bytes[..1])? as u32,
                    DiscriminantSize::Size2 => u16::load(mem, &bytes[..2])? as u32,
                    DiscriminantSize::Size4 => u32::load(mem, &bytes[..4])?,
                };
                let case_ty = types.get(discriminant as usize).ok_or_else(|| {
                    anyhow!(
                        "discriminant {} out of range [0..{})",
                        discriminant,
                        types.len()
                    )
                })?;
                let value = Rc::new(Self::load(
                    store,
                    mem,
                    case_ty,
                    &bytes[func::align_to(
                        usize::from(discriminant_size),
                        SizeAndAlignment::from(ty).alignment,
                    )..][..SizeAndAlignment::from(case_ty).size],
                )?);

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            Type::Flags(count) => Val::Flags {
                count: u32::try_from(*count)?,
                value: match FlagsSize::from_count(*count) {
                    FlagsSize::Size1 => iter::once(u8::load(mem, bytes)? as u32).collect(),
                    FlagsSize::Size2 => iter::once(u16::load(mem, bytes)? as u32).collect(),
                    FlagsSize::Size4Plus(n) => (0..n)
                        .map(|index| u32::load(mem, &bytes[index * 4..][..4]))
                        .collect::<Result<_>>()?,
                },
            },
        })
    }
}

/// Lower a list with the specified element type and values.
fn lower_list<T>(
    mem: &mut MemoryMut<'_, T>,
    element_type: &Type,
    items: &[Val],
) -> Result<(usize, usize)> {
    let SizeAndAlignment {
        size: element_size,
        alignment: element_alignment,
    } = SizeAndAlignment::from(element_type);
    let size = items
        .len()
        .checked_mul(element_size)
        .ok_or_else(|| anyhow::anyhow!("size overflow copying a list"))?;
    let ptr = mem.realloc(0, 0, element_alignment, size)?;
    let mut element_ptr = ptr;
    for item in items {
        item.store(mem, element_type, element_ptr)?;
        element_ptr += element_size;
    }
    Ok((ptr, items.len()))
}

/// Represents the size and alignment requirements of the heap-serialized form of a type
pub(crate) struct SizeAndAlignment {
    pub(crate) size: usize,
    pub(crate) alignment: u32,
}

impl SizeAndAlignment {
    /// Calculate the size and alignment requirements for the specified type.
    pub(crate) fn from(ty: &Type) -> Self {
        match ty {
            Type::Bool | Type::S8 | Type::U8 => Self {
                size: 1,
                alignment: 1,
            },

            Type::S16 | Type::U16 => Self {
                size: 2,
                alignment: 2,
            },

            Type::S32 | Type::U32 | Type::Char | Type::Float32 => Self {
                size: 4,
                alignment: 4,
            },

            Type::S64 | Type::U64 | Type::Float64 => Self {
                size: 8,
                alignment: 8,
            },

            Type::String | Type::List(_) => Self {
                size: 8,
                alignment: 4,
            },

            Type::Record(types) => {
                let mut offset = 0;
                let mut align = 1;
                for ty in types.iter() {
                    let SizeAndAlignment { size, alignment } = Self::from(ty);
                    offset = func::align_to(offset, alignment) + size;
                    align = align.max(alignment);
                }

                Self {
                    size: func::align_to(offset, align),
                    alignment: align,
                }
            }

            Type::Variant(types) => {
                let discriminant_size = DiscriminantSize::from_count(types.len()).unwrap();
                let mut alignment = 1;
                let mut size = 0;
                for ty in types.iter() {
                    let s_and_a = Self::from(ty);
                    alignment = alignment.max(s_and_a.alignment);
                    size = size.max(s_and_a.size);
                }

                Self {
                    size: func::align_to(usize::from(discriminant_size), alignment) + size,
                    alignment,
                }
            }

            Type::Flags(count) => match FlagsSize::from_count(*count) {
                FlagsSize::Size1 => Self {
                    size: 1,
                    alignment: 1,
                },
                FlagsSize::Size2 => Self {
                    size: 2,
                    alignment: 2,
                },
                FlagsSize::Size4Plus(n) => Self {
                    size: n * 4,
                    alignment: 4,
                },
            },
        }
    }
}

/// Calculate the aligned offset of a field of the specified type, updating `offset` to point to just after that
/// field.
pub(crate) fn next_field(ty: &Type, offset: &mut usize) -> usize {
    let SizeAndAlignment { size, alignment } = SizeAndAlignment::from(ty);
    *offset = func::align_to(*offset, alignment);
    let result = *offset;
    *offset += size;
    result
}

/// Calculate the size of a u32 array needed to represent the specified number of bit flags.
///
/// Note that this will always return at least 1, even if the `count` parameter is zero.
fn u32_count_for_flag_count(count: usize) -> usize {
    match FlagsSize::from_count(count) {
        FlagsSize::Size1 | FlagsSize::Size2 => 1,
        FlagsSize::Size4Plus(n) => n,
    }
}
