use crate::component::func::{self, DiscriminantSize, FlagsSize, Lift, MemoryMut, Options};
use crate::store::StoreOpaque;
use crate::{AsContextMut, StoreContextMut, ValRaw};
use anyhow::{anyhow, bail, Result};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;
use wasmtime_environ::component::{ComponentTypes, InterfaceType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Val {
    Bool(bool),
    S8(i8),
    U8(u8),
    S16(i16),
    U16(u16),
    S32(i32),
    U32(u32),
    S64(i64),
    U64(u64),
    Float32(u32),
    Float64(u64),
    Char(char),
    String(Rc<str>),
    List(Rc<[Val]>),
    Record(Rc<[Val]>),
    Variant { discriminant: u32, value: Rc<Val> },
    Flags { count: u32, value: Rc<[u32]> },
}

impl Val {
    pub fn typecheck(&self, ty: &InterfaceType, types: &ComponentTypes) -> Result<()> {
        match (self, ty) {
            (Val::Bool(_), InterfaceType::Bool)
            | (Val::S8(_), InterfaceType::S8)
            | (Val::U8(_), InterfaceType::U8)
            | (Val::S16(_), InterfaceType::S16)
            | (Val::U16(_), InterfaceType::U16)
            | (Val::S32(_), InterfaceType::S32)
            | (Val::U32(_), InterfaceType::U32)
            | (Val::S64(_), InterfaceType::S64)
            | (Val::U64(_), InterfaceType::U64)
            | (Val::Float32(_), InterfaceType::Float32)
            | (Val::Float64(_), InterfaceType::Float64)
            | (Val::Char(_), InterfaceType::Char)
            | (Val::String(_), InterfaceType::String) => (),

            (Val::List(items), InterfaceType::List(index)) => {
                let ty = &types[*index];

                for item in items.deref() {
                    item.typecheck(ty, types)?;
                }
            }

            (Val::Record(values), InterfaceType::Record(index)) => {
                let fields = &types[*index].fields;

                if fields.len() != values.len() {
                    bail!(
                        "expected {} field values, got {}",
                        fields.len(),
                        values.len()
                    );
                }

                for (field, value) in fields.iter().zip(values.deref()) {
                    value.typecheck(&field.ty, types)?;
                }
            }

            (Val::Record(values), InterfaceType::Tuple(index)) => {
                let tuple_types = &types[*index].types;

                if tuple_types.len() != values.len() {
                    bail!(
                        "expected {} values, got {}",
                        tuple_types.len(),
                        values.len()
                    );
                }

                for (ty, value) in tuple_types.iter().zip(values.deref()) {
                    value.typecheck(ty, types)?;
                }
            }

            (Val::Record(values), InterfaceType::Unit) => {
                if !values.is_empty() {
                    bail!("expected 0 values, got {}", values.len());
                }
            }

            (
                Val::Variant {
                    discriminant,
                    value,
                },
                InterfaceType::Variant(index),
            ) => {
                let cases = &types[*index].cases;

                if *discriminant as usize >= cases.len() {
                    bail!(
                        "discriminant {} is out of expected range [0, {})",
                        discriminant,
                        cases.len()
                    );
                }

                value.typecheck(&cases[*discriminant as usize].ty, types)?;
            }

            (
                Val::Variant {
                    discriminant,
                    value,
                },
                InterfaceType::Enum(index),
            ) => {
                let names = &types[*index].names;

                if *discriminant as usize >= names.len() {
                    bail!(
                        "discriminant {} is out of expected range [0, {})",
                        discriminant,
                        names.len()
                    );
                }

                value.typecheck(&InterfaceType::Unit, types)?;
            }

            (
                Val::Variant {
                    discriminant,
                    value,
                },
                InterfaceType::Union(index),
            ) => {
                let union_types = &types[*index].types;

                if *discriminant as usize >= union_types.len() {
                    bail!(
                        "discriminant {} is out of expected range [0, {})",
                        discriminant,
                        union_types.len()
                    );
                }

                value.typecheck(&union_types[*discriminant as usize], types)?;
            }

            (
                Val::Variant {
                    discriminant,
                    value,
                },
                InterfaceType::Option(index),
            ) => match discriminant {
                0 => value.typecheck(&InterfaceType::Unit, types)?,
                1 => value.typecheck(&types[*index], types)?,
                _ => bail!(
                    "discriminant {} is out of expected range [0, 2)",
                    discriminant
                ),
            },

            (
                Val::Variant {
                    discriminant,
                    value,
                },
                InterfaceType::Expected(index),
            ) => match discriminant {
                0 => value.typecheck(&types[*index].ok, types)?,
                1 => value.typecheck(&types[*index].err, types)?,
                _ => bail!(
                    "discriminant {} is out of expected range [0, 2)",
                    discriminant
                ),
            },

            (Val::Flags { count, value }, InterfaceType::Flags(index)) => {
                let count = *count as usize;
                let names = &types[*index].names;

                if count > value.len() * 32 {
                    bail!(
                        "flag count {} must not be larger than value count {}",
                        count,
                        value.len() * 32
                    );
                }

                if func::ceiling_divide(count, 32) > value.len() {
                    bail!(
                        "value count {} must not be larger than required by flag count {}",
                        value.len() * 32,
                        count,
                    );
                }

                if names.len() != count {
                    bail!("expected {} flags, got {}", names.len(), count);
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
            Val::Variant { .. } => "variant",
            Val::Flags { .. } => "flags",
        }
    }

    pub(crate) fn lower<T>(
        &self,
        store: &mut StoreContextMut<T>,
        options: &Options,
        ty: &InterfaceType,
        types: &ComponentTypes,
        vec: &mut Vec<ValRaw>,
    ) -> Result<()> {
        match self {
            Val::Bool(value) => vec.push(ValRaw::u32(if *value { 1 } else { 0 })),
            Val::S8(value) => vec.push(ValRaw::i32(*value as i32)),
            Val::U8(value) => vec.push(ValRaw::u32(*value as u32)),
            Val::S16(value) => vec.push(ValRaw::i32(*value as i32)),
            Val::U16(value) => vec.push(ValRaw::u32(*value as u32)),
            Val::S32(value) => vec.push(ValRaw::i32(*value)),
            Val::U32(value) => vec.push(ValRaw::u32(*value)),
            Val::S64(value) => vec.push(ValRaw::i64(*value)),
            Val::U64(value) => vec.push(ValRaw::u64(*value)),
            Val::Float32(value) => vec.push(ValRaw::f32(*value)),
            Val::Float64(value) => vec.push(ValRaw::f64(*value)),
            Val::Char(value) => vec.push(ValRaw::u32(u32::from(*value))),
            Val::String(value) => {
                let (ptr, len) = super::lower_string(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    value,
                )?;
                vec.push(ValRaw::i64(ptr as i64));
                vec.push(ValRaw::i64(len as i64));
            }
            Val::List(items) => {
                let (ptr, len) = lower_list(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    ty,
                    types,
                    &items,
                )?;
                vec.push(ValRaw::i64(ptr as i64));
                vec.push(ValRaw::i64(len as i64));
            }
            Val::Record(values) => {
                for (value, ty) in values.iter().zip(record_types(ty, types)) {
                    value.lower(store, options, ty, types, vec)?;
                }
            }
            Val::Variant {
                discriminant,
                value,
            } => {
                vec.push(ValRaw::u32(*discriminant));
                value.lower(
                    store,
                    options,
                    variant_types(ty, types)[*discriminant as usize],
                    types,
                    vec,
                )?;
            }
            Val::Flags { count, value } => {
                vec.extend(value.iter().map(|&v| ValRaw::u32(v)));
            }
        }

        Ok(())
    }

    pub(crate) fn store<T>(
        &self,
        mem: &mut MemoryMut<'_, T>,
        ty: &InterfaceType,
        types: &ComponentTypes,
        offset: usize,
    ) -> Result<()> {
        match self {
            Val::Bool(value) => value.store(mem, offset)?,
            Val::S8(value) => value.store(mem, offset)?,
            Val::U8(value) => value.store(mem, offset)?,
            Val::S16(value) => value.store(mem, offset)?,
            Val::U16(value) => value.store(mem, offset)?,
            Val::S32(value) => value.store(mem, offset)?,
            Val::U32(value) => value.store(mem, offset)?,
            Val::S64(value) => value.store(mem, offset)?,
            Val::U64(value) => value.store(mem, offset)?,
            Val::Float32(value) => value.store(mem, offset)?,
            Val::Float64(value) => value.store(mem, offset)?,
            Val::Char(value) => value.store(mem, offset)?,
            Val::String(value) => {
                let (ptr, len) = super::lower_string(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    value,
                )?;
                // FIXME: needs memory64 handling
                *mem.get(offset + 0) = (ptr as i32).to_le_bytes();
                *mem.get(offset + 4) = (len as i32).to_le_bytes();
            }
            Val::List(items) => {
                let (ptr, len) = lower_list(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    ty,
                    types,
                    &items,
                )?;
                // FIXME: needs memory64 handling
                *mem.get(offset + 0) = (ptr as i32).to_le_bytes();
                *mem.get(offset + 4) = (len as i32).to_le_bytes();
            }
            Val::Record(values) => {
                let mut offset = offset;
                for (value, ty) in values.iter().zip(record_types(ty, types)) {
                    value.store(mem, ty, types, next_field(ty, types, &mut offset))?;
                }
            }
            Val::Variant {
                discriminant,
                value,
            } => {
                let types = variant_types(ty, types);
                let case_ty = types[*discriminant as usize];
                let mut offset = offset;
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
                    types,
                    offset
                        + align_to(
                            discriminant_size.into(),
                            SizeAndAlignment::from(ty, types).alignment,
                        ),
                )?;
            }
            Val::Flags { count, value } => match FlagsSize::from_count(count) {
                FlagsSize::Size1 => u8::try_from(value[0]).unwrap().store(mem, offset)?,
                FlagsSize::Size2 => u16::try_from(value[0]).unwrap().store(mem, offset)?,
                FlagsSize::Size4Plus(_) => {
                    let mut offset = offset;
                    for value in value.deref() {
                        value.store(mem, offset)?;
                        offset += 4;
                    }
                }
            },
        }

        Ok(())
    }

    pub(crate) fn lift<'a>(
        store: &StoreOpaque,
        options: &Options,
        ty: &InterfaceType,
        types: &ComponentTypes,
        src: &mut impl Iterator<Item = &'a ValRaw>,
    ) -> Result<Self> {
        fn next<'a>(src: &mut impl Iterator<Item = &'a ValRaw>) -> &'a ValRaw {
            src.next().unwrap()
        }

        Ok(match ty {
            InterfaceType::Bool => Val::Bool(bool::lift(store, options, next(src))?),
            InterfaceType::S8 => Val::S8(i8::lift(store, options, next(src))?),
            InterfaceType::U8 => Val::U8(u8::lift(store, options, next(src))?),
            InterfaceType::S16 => Val::S16(i16::lift(store, options, next(src))?),
            InterfaceType::U16 => Val::U16(u16::lift(store, options, next(src))?),
            InterfaceType::S32 => Val::S32(i32::lift(store, options, next(src))?),
            InterfaceType::U32 => Val::U32(u32::lift(store, options, next(src))?),
            InterfaceType::S64 => Val::S64(i64::lift(store, options, next(src))?),
            InterfaceType::U64 => Val::U64(u64::lift(store, options, next(src))?),
            InterfaceType::Float32 => Val::Float32(u32::lift(store, options, next(src))?),
            InterfaceType::Float64 => Val::Float64(u64::lift(store, options, next(src))?),
            InterfaceType::Char => Val::Char(char::lift(store, options, next(src))?),
            InterfaceType::String | InterfaceType::List(_) => {
                // These won't fit in func::MAX_STACK_RESULTS as of this writing, so presumably we should never
                // reach here
                unreachable!()
            }
            InterfaceType::Record(_) | InterfaceType::Tuple(_) | InterfaceType::Unit => {
                Val::Record(
                    record_types(ty, types)
                        .into_iter()
                        .map(|ty| Self::lift(store, options, ty, types, src))
                        .collect::<Result<_>>()?,
                )
            }
            InterfaceType::Variant(_)
            | InterfaceType::Enum(_)
            | InterfaceType::Union(_)
            | InterfaceType::Optional(_)
            | InterfaceType::Expected(_) => {
                let types = variant_types(ty, types);
                let discriminant = next(src).get_u32();
                let case_ty = types.get(discriminant as usize).ok_or_else(|| {
                    anyhow!(
                        "discriminant {} out of range [0..{})",
                        discriminant,
                        types.len()
                    )
                })?;
                let value = Rc::new(Self::lift(store, options, case_ty, types, src)?);

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            InterfaceType::Flags(index) => {
                let names = &types[*index].names;
                let count = u32::try_from(names.len()).unwrap();
                assert!(count <= 32);
                let value = iter::once(u32::lift(store, options, next(src))?).collect();

                Val::Flags { count, value }
            }
        })
    }

    fn load(
        store: &StoreOpaque,
        mem: &Memory,
        ty: &InterfaceType,
        types: &ComponentTypes,
        bytes: &[u8],
    ) -> Result<Self> {
        Ok(match ty {
            InterfaceType::Bool => Val::Bool(bool::load(mem, bytes)?),
            InterfaceType::S8 => Val::S8(i8::load(mem, bytes)?),
            InterfaceType::U8 => Val::U8(u8::load(mem, bytes)?),
            InterfaceType::S16 => Val::S16(i16::load(mem, bytes)?),
            InterfaceType::U16 => Val::U16(u16::load(mem, bytes)?),
            InterfaceType::S32 => Val::S32(i32::load(mem, bytes)?),
            InterfaceType::U32 => Val::U32(u32::load(mem, bytes)?),
            InterfaceType::S64 => Val::S64(i64::load(mem, bytes)?),
            InterfaceType::U64 => Val::U64(u64::load(mem, bytes)?),
            InterfaceType::Float32 => Val::Float32(u32::load(mem, bytes)?),
            InterfaceType::Float64 => Val::Float64(u64::load(mem, bytes)?),
            InterfaceType::Char => Val::Char(char::load(mem, bytes)?),
            InterfaceType::String => {
                Val::String(Rc::from(WasmStr::load(mem, bytes)?.to_str(store)?.deref()))
            }
            InterfaceType::List(index) => {
                let element_type = &types[*index];
                // FIXME: needs memory64 treatment
                let ptr = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                let len = u32::from_le_bytes(bytes[4..].try_into().unwrap());
                let SizeAndAlignment {
                    size: element_size,
                    alignment: element_alignment,
                } = SizeAndAlignment::from(element_type, types);

                match len
                    .checked_mul(element_size)
                    .and_then(|len| ptr.checked_add(len))
                {
                    Some(n) if n <= memory.as_slice().len() => {}
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
                                types,
                                &memory.as_slice()[ptr + (index * element_size)..][..element_size],
                            )
                        })
                        .collect::<Result<_>>(),
                )
            }
            InterfaceType::Record(_) | InterfaceType::Tuple(_) | InterfaceType::Unit => {
                Val::Record(
                    record_types(ty, types)
                        .into_iter()
                        .map(|ty| {
                            Self::load(
                                store,
                                options,
                                ty,
                                types,
                                &bytes[next_field(ty, &mut offset)..]
                                    [..SizeAndAlignment::from(ty, types).size],
                            )
                        })
                        .collect::<Result<_>>()?,
                )
            }
            InterfaceType::Variant(_)
            | InterfaceType::Enum(_)
            | InterfaceType::Union(_)
            | InterfaceType::Optional(_)
            | InterfaceType::Expected(_) => {
                let types = variant_types(ty, types);
                let discriminant_size = DiscriminantSize::from_count(types.len()).unwrap();
                let discriminant = match discriminant_size {
                    DiscriminantSize::Size1 => u8::load(mem, &bytes[..1])? as u32,
                    DiscriminantSize::Size2 => u16::load(mem, &bytes[..2])? as u32,
                    DiscriminantSize::Size1 => u32::load(mem, &bytes[..4])?,
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
                    types,
                    &bytes[align_to(
                        usize::from(discriminant_size),
                        SizeAndAlignment::from(ty, types).alignment,
                    )..][..SizeAndAlignment::from(case_ty, types).size],
                )?);

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            InterfaceType::Flags(index) => {
                Val::Flags(match FlagsSize::from_count(types[*index].names) {
                    FlagsSize::Size1 => iter::once(u8::load(mem, bytes)? as u32).collect(),
                    FlagsSize::Size2 => iter::once(u16::load(mem, bytes)? as u32).collect(),
                    FlagsSize::Size4Plus(n) => (0..n)
                        .map(|index| u32::load(mem, bytes[index * 4..][..4]))
                        .collect::<Result<_>>()?,
                })
            }
        })
    }
}

fn record_types(ty: &InterfaceType, types: &ComponentTypes) -> Vec<&InterfaceType> {
    match ty {
        InterfaceType::Record(index) => types[*index]
            .fields
            .iter()
            .map(|field| &field.ty)
            .collect::<Vec<_>>(),
        InterfaceType::Tuple(index) => types[*index].types.iter().collect::<Vec<_>>(),
        InterfaceType::Unit => Vec::new(),
        _ => unreachable!(),
    }
}

fn variant_types(
    ty: &InterfaceType,
    types: &ComponentTypes,
    discriminant: usize,
) -> Vec<&InterfaceType> {
    match ty {
        InterfaceType::Variant(index) => types[*index]
            .cases
            .iter()
            .map(|case| &case.ty)
            .collect::<Vec<_>>(),
        InterfaceType::Enum(index) => types[*index]
            .names
            .iter()
            .map(|_| &InterfaceType::Unit)
            .collect(),
        InterfaceType::Union(index) => types[*index].types.iter().collect(),
        InterfaceType::Optional(index) => {
            vec![&InterfaceType::Unit, &types[*index]]
        }
        InterfaceType::Expected(index) => {
            let cases = &types[*index];
            vec![&cases.ok, &cases.err]
        }
        _ => unreachable!(),
    }
}

fn lower_list<T>(
    mem: &mut MemoryMut<'_, T>,
    ty: &InterfaceType,
    types: &ComponentTypes,
    vec: &[Val],
) -> Result<(usize, usize)> {
    let element_type = if let InterfaceType::List(index) = ty {
        &types[*index]
    } else {
        unreachable!()
    };
    let memory = &mut MemoryMut::new(store.as_context_mut(), options);
    let SizeAndAlignment {
        size: element_size,
        alignment: element_alignment,
    } = SizeAndAlignment::from(element_type, types);
    let size = items
        .len()
        .checked_mul(element_size)
        .ok_or_else(|| anyhow::anyhow!("size overflow copying a list"))?;
    let ptr = memory.realloc(0, 0, element_alignment, size)?;
    let mut element_ptr = ptr;
    for item in items {
        item.store(memory, ty, types, element_ptr)?;
        element_ptr += element_size;
    }
    (ptr, items.len())
}

pub(crate) fn flatten_count(ty: &InterfaceType, types: &ComponentTypes) -> usize {
    match ty {
        InterfaceType::Unit => 0,
        InterfaceType::Bool
        | InterfaceType::S8
        | InterfaceType::U8
        | InterfaceType::S16
        | InterfaceType::U16
        | InterfaceType::S32
        | InterfaceType::U32
        | InterfaceType::S64
        | InterfaceType::U64
        | InterfaceType::Float32
        | InterfaceType::Float64
        | InterfaceType::Char
        | InterfaceType::Enum(_) => mem::size_of::<ValRaw>(),

        InterfaceType::String => mem::size_of::<ValRaw>() * 2,

        InterfaceType::Record(index) => types[*index]
            .fields
            .iter()
            .map(|field| flatten_count(&field.ty, types))
            .sum(),

        InterfaceType::Variant(index) => {
            mem::size_of::<ValRaw>()
                + types[*index]
                    .cases
                    .iter()
                    .map(|case| flatten_count(&case.ty, types))
                    .max()
        }

        InterfaceType::List(_) => mem::size_of::<ValRaw>() * 2,

        InterfaceType::Tuple(index) => types[*index]
            .types
            .iter()
            .map(|ty| flatten_count(ty, types))
            .sum(),

        InterfaceType::Flags(index) => {
            mem::size_of::<ValRaw>() * func::ceiling_divide(types[*index].names, 32).max(1)
        }

        InterfaceType::Union(index) => {
            mem::size_of::<ValRaw>()
                + types[*index]
                    .types
                    .iter()
                    .map(|ty| flatten_count(ty, types))
                    .max()
        }

        InterfaceType::Option(index) => {
            mem::size_of::<ValRaw>() + flatten_count(&types[*index], types)
        }

        InterfaceType::Expected(index) => {
            mem::size_of::<ValRaw>()
                + flatten_count(&types[*index].ok, types)
                    .max(flatten_count(&types[*index].err, types))
        }
    }
}

pub(crate) struct SizeAndAlignment {
    pub(crate) size: usize,
    pub(crate) alignment: u32,
}

impl SizeAndAlignment {
    pub(crate) fn from(ty: &InterfaceType, types: &ComponentTypes) -> Self {
        match ty {
            InterfaceType::Unit => Self {
                size: 0,
                alignment: 1,
            },
            InterfaceType::Bool | InterfaceType::S8 | InterfaceType::U8 => Self {
                size: 1,
                alignment: 1,
            },
            InterfaceType::S16 | InterfaceType::U16 => Self {
                size: 2,
                alignment: 2,
            },
            InterfaceType::S32
            | InterfaceType::U32
            | InterfaceType::Char
            | InterfaceType::Float32 => Self {
                size: 4,
                alignment: 4,
            },
            InterfaceType::S64 | InterfaceType::U64 | InterfaceType::Float64 => Self {
                size: 8,
                alignment: 8,
            },

            InterfaceType::Enum(index) => {
                let discriminant_size =
                    DiscriminantSize::from_count(types[*index].names.len()).unwrap();

                Self {
                    size: discriminant_size.into(),
                    alignment: discriminant_size.into(),
                }
            }

            InterfaceType::String | InterfaceType::List(_) => Self {
                size: 8,
                alignment: 4,
            },

            InterfaceType::Record(index) => {
                let mut offset = 0;
                let mut align = 1;
                for field in &types[*index].fields {
                    let SizeAndAlignment { size, alignment } = Self::from(&field.ty, types);
                    offset = align_to(offset, alignment) + size;
                    align = align.max(alignment);
                }

                Self {
                    size: offset,
                    alignment: align,
                }
            }

            InterfaceType::Variant(index) => {
                let cases = &types[*index].cases;
                let discriminant_size =
                    usize::from(DiscriminantSize::from_count(cases.len()).unwrap());
                let alignment = discriminant_size.max(
                    cases
                        .iter()
                        .map(|case| Self::from(&case.ty, types).alignment)
                        .max(),
                );

                Self {
                    size: align_to(discriminant_size, alignment)
                        + cases
                            .iter()
                            .map(|case| Self::from(&case.ty, types).size)
                            .max(),
                    alignment,
                }
            }

            InterfaceType::Tuple(index) => {
                let mut offset = 0;
                let mut align = 1;
                for ty in &types[*index].types {
                    let SizeAndAlignment { size, alignment } = Self::from(ty, types);
                    offset = align_to(offset, alignment) + size;
                    align = align.max(alignment);
                }

                Self {
                    size: offset,
                    alignment: align,
                }
            }

            InterfaceType::Flags(index) => match FlagsSize::from_count(types[*index].names.len()) {
                FlagsSize::Size1 => Self {
                    size: 1,
                    alignment: 1,
                },
                FlagsSize::Size2 => Self {
                    size: 2,
                    alignment: 2,
                },
                FlgasSize::Size4Plus(n) => Self {
                    size: n * 4,
                    alignment: 4,
                },
            },

            InterfaceType::Union(index) => {
                let types = &types[*index].types;
                let discriminant_size =
                    usize::from(DiscriminantSize::from_count(types.len()).unwrap());
                let alignment = discriminant_size
                    .max(types.iter().map(|ty| Self::from(ty, types).alignment).max());

                Self {
                    size: align_to(discriminant_size, alignment)
                        + types.iter().map(|tu| Self::from(ty, types).size).max(),
                    alignment,
                }
            }

            InterfaceType::Option(index) => {
                let SizeAndAlignment { size, alignment } = Self::from(&types[*index], types);

                Self {
                    size: align_to(1, alignment) + size,
                    alignment,
                }
            }

            InterfaceType::Expected(index) => {
                let SizeAndAlignment {
                    ok_size,
                    ok_alignment,
                } = Self::from(&types[*index].ok, types);

                let SizeAndAlignment {
                    err_size,
                    err_alignment,
                } = Self::from(&types[*index].err, types);

                Self {
                    size: align_to(1, alignment) + ok_size.max(err_size),
                    alignment: ok_alignment.max(err_alignment),
                }
            }
        }
    }
}

pub fn next_field(ty: &InterfaceType, types: &ComponentTypes, offset: &mut usize) -> usize {
    let SizeAndAlignment { size, alignment } = SizeAndAlignment::from(ty, types);
    *offset = align_to(*offset, alignment);
    let result = *offset;
    *offset += size;
    result
}
