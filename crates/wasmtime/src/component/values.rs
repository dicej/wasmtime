use crate::component::func::{self, Lift, Lower, Memory, MemoryMut, Options, WasmStr};
use crate::store::StoreOpaque;
use crate::{AsContextMut, StoreContextMut, ValRaw};
use anyhow::{anyhow, bail, Result};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;
use wasmtime_component_util::{DiscriminantSize, FlagsSize};
use wasmtime_environ::component::{ComponentTypes, InterfaceType};

pub struct List {
    ty: Handle<TypeIndex>,
    values: Box<[Val]>,
}

pub struct Record {
    ty: Handle<RecordIndex>,
    values: Box<[Val]>,
}

pub struct Variant {
    ty: Handle<VariantIndex>,
    discriminant: u32,
    value: Box<Val>,
}

pub struct Flags {
    ty: Handle<FlagsIndex>,
    count: u32,
    value: Box<[u32]>,
}

pub struct Tuple {
    ty: Handle<TupleIndex>,
    values: Box<[Val]>,
}

pub struct Enum {
    ty: Handle<EnumIndex>,
    discriminant: u32,
}

pub struct Union {
    ty: Handle<UnionIndex>,
    discriminant: u32,
    value: Box<Val>,
}

pub struct Option {
    ty: Handle<TypeIndex>,
    discriminant: u32,
    value: Box<Val>,
}

pub struct Expected {
    ty: Handle<ExpectedIndex>,
    discriminant: u32,
    value: Box<Val>,
}

/// Possible runtime values which a component function can either consume or produce
#[derive(Debug, PartialEq, Eq)]
pub enum Val {
    Unit,
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
    String(Box<str>),
    /// List of values
    List(List),
    /// Record, tuple, or unit
    Record(Record),
    /// Variant, enum, or union
    Variant(Variant),
    /// Bit flags
    Flags(Flags),
    Tuple(Tuple),
    Enum(Enum),
    Union(Union),
    Option(Option),
    Expected(Expected),
}

impl Val {
    pub fn ty(&self) -> Type {
        match self {
            Val::Unit => Type::Unit,
            Val::Bool(_) => Type::Bool,
            Val::S8(_) => Type::S8,
            Val::U8(_) => Type::U8,
            Val::S16(_) => Type::S16,
            Val::U16(_) => Type::U16,
            Val::S32(_) => Type::S32,
            Val::U32(_) => Type::U32,
            Val::S64(_) => Type::S64,
            Val::U64(_) => Type::U64,
            Val::Float32(_) => Type::Float32,
            Val::Float64(_) => Type::Float64,
            Val::Char(_) => Type::Char,
            Val::String(_) => Type::String,
            Val::List(List { ty, .. }) => Type::List(ty),
            Val::Record(Record { ty, .. }) => Type::Record(ty),
            Val::Variant(Variant { ty, .. }) => Type::Variant(ty),
            Val::Flags(Flags { ty, .. }) => Type::Flags(ty),
            Val::Tuple(Tuple { ty, .. }) => Type::Tuple(ty),
            Val::Enum(Enum { ty, .. }) => Type::Enum(ty),
            Val::Union(Union { ty, .. }) => Type::Union(ty),
            Val::Option(Option { ty, .. }) => Type::Option(ty),
            Val::Expected(Expected { ty, .. }) => Type::Expected(ty),
        }
    }

    /// Serialize this value as core Wasm stack values.
    pub(crate) fn lower<T>(
        &self,
        store: &mut StoreContextMut<T>,
        options: &Options,
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
            Val::List(List { values, .. }) => {
                let (ptr, len) =
                    lower_list(&mut MemoryMut::new(store.as_context_mut(), options), values)?;
                vec.push(ValRaw::i64(ptr as i64));
                vec.push(ValRaw::i64(len as i64));
            }
            Val::Record(Record { values, .. }) | Val::Tuple(Tuple { values, .. }) => {
                for value in values.deref() {
                    value.lower(store, options, vec)?;
                }
            }
            Val::Variant(Variant {
                discriminant,
                value,
                ..
            })
            | Val::Union(Union {
                discriminant,
                value,
                ..
            })
            | Val::Option(Option {
                discriminant,
                value,
                ..
            })
            | Val::Expected(Expected {
                discriminant,
                value,
                ..
            }) => {
                vec.push(ValRaw::u32(*discriminant));
                value.lower(store, options, vec)?;
                // Pad `vec` out to max payload size:
                vec.extend(
                    iter::repeat(ValRaw::u32(0))
                        .take(self.ty().flatten_count().checked_sub(vec.len()).unwrap()),
                );
            }
            Val::Enum(Enum { discriminant, .. }) => {
                vec.push(ValRaw::u32(*discriminant));
            }
            Val::Flags(Flags { value, .. }) => {
                vec.extend(value.iter().map(|&v| ValRaw::u32(v)));
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Serialize this value to the heap at the specified memory location.
    pub(crate) fn store<T>(&self, mem: &mut MemoryMut<'_, T>, offset: usize) -> Result<()> {
        match (self, ty) {
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
                let (ptr, len) = super::lower_string(mem, value)?;
                // FIXME: needs memory64 handling
                *mem.get(offset + 0) = (ptr as i32).to_le_bytes();
                *mem.get(offset + 4) = (len as i32).to_le_bytes();
            }
            Val::List(List { values, .. }) => {
                let (ptr, len) = lower_list(mem, values)?;
                // FIXME: needs memory64 handling
                *mem.get(offset + 0) = (ptr as i32).to_le_bytes();
                *mem.get(offset + 4) = (len as i32).to_le_bytes();
            }
            Val::Record(Record { values, .. }) | Val::Tuple(Tuple { values, .. }) => {
                let mut offset = offset;
                for value in values.deref() {
                    value.store(mem, next_field(&value.ty(), &mut offset))?;
                }
            }
            Val::Variant(Variant {
                discriminant,
                value,
                ty,
            }) => store_variant(discriminant, value, ty.cases().len()),

            Val::Union(Union {
                discriminant,
                value,
                ty,
            }) => store_variant(discriminant, value, ty.types().len()),

            Val::Option(Option {
                discriminant,
                value,
                ..
            })
            | Val::Expected(Expected {
                discriminant,
                value,
                ..
            }) => store_variant(discriminant, value, 2),

            Val::Flags(Flags { count, value, .. }) => {
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
}

fn store_variant(discriminant: u32, value: &Value, case_count: usize) -> Result<()> {
    let discriminant_size = DiscriminantSize::from_count(case_count).unwrap();
    match discriminant_size {
        DiscriminantSize::Size1 => u8::try_from(*discriminant).unwrap().store(mem, offset)?,
        DiscriminantSize::Size2 => u16::try_from(*discriminant).unwrap().store(mem, offset)?,
        DiscriminantSize::Size4 => (*discriminant).store(mem, offset)?,
    }

    value.store(
        mem,
        offset + func::align_to(discriminant_size.into(), ty.size_and_alignment().alignment),
    )?;
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
    } = element_type.size_and_alignment();
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

/// Calculate the aligned offset of a field of the specified type, updating `offset` to point to just after that
/// field.
pub(crate) fn next_field(ty: &Type, offset: &mut usize) -> usize {
    let SizeAndAlignment { size, alignment } = ty.size_and_alignment();
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
