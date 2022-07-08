use crate::component::func::{Lift, MemoryMut, Options};
use crate::store::StoreOpaque;
use crate::{AsContextMut, StoreContextMut, ValRaw};
use anyhow::{anyhow, bail, Result};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;
use wasmtime_environ::component::{ComponentTypes, InterfaceType};

fn ceiling_divide(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

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

                if ceiling_divide(count, 32) > value.len() {
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
        vec: &mut Vec<ValRaw>,
    ) -> Result<()> {
        Ok(match self {
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
                let (ptr, len) = super::lower_list(
                    &mut MemoryMut::new(store.as_context_mut(), options),
                    value.deref(),
                )?;
                vec.push(ValRaw::i64(ptr as i64));
                vec.push(ValRaw::i64(len as i64));
            }
            Val::Record(values) => {
                for value in values.deref() {
                    value.lower(store, options, vec)?;
                }
            }
            Val::Variant {
                discriminant,
                value,
            } => {
                vec.push(ValRaw::u32(*discriminant));
                value.lower(store, options, vec)?;
            }
            Val::Flags { count, value } => {
                vec.extend(value.iter().map(|&v| ValRaw::u32(v)));
            }
        })
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
            InterfaceType::Unit => Val::Record(Rc::new([])),
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
            InterfaceType::String => unreachable!(),
            InterfaceType::Record(index) => {
                let fields = &types[*index].fields;

                Val::Record(
                    fields
                        .iter()
                        .map(|field| Self::lift(store, options, &field.ty, types, src))
                        .collect::<Result<_>>()?,
                )
            }
            InterfaceType::Variant(index) => {
                let cases = &types[*index].cases;
                let discriminant = next(src).get_u32();
                let case = cases.get(discriminant as usize).ok_or_else(|| {
                    anyhow!(
                        "discriminant {} out of range [0..{})",
                        discriminant,
                        cases.len()
                    )
                })?;
                let value = Rc::new(Self::lift(store, options, &case.ty, types, src)?);

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            InterfaceType::List(_) => unreachable!(),
            InterfaceType::Tuple(index) => {
                let tuple_types = &types[*index].types;

                Val::Record(
                    tuple_types
                        .iter()
                        .map(|ty| Self::lift(store, options, ty, types, src))
                        .collect::<Result<_>>()?,
                )
            }
            InterfaceType::Flags(index) => {
                let names = &types[*index].names;
                let count = u32::try_from(names.len()).unwrap();
                assert!(count <= 32);
                let value = iter::once(u32::lift(store, options, next(src))?).collect();

                Val::Flags { count, value }
            }
            InterfaceType::Enum(index) => {
                let names = &types[*index].names;
                let discriminant = next(src).get_u32();
                names.get(discriminant as usize).ok_or_else(|| {
                    anyhow!(
                        "discriminant {} out of range [0..{})",
                        discriminant,
                        names.len()
                    )
                })?;

                Val::Variant {
                    discriminant,
                    value: Rc::new(Val::Record(Rc::new([]))),
                }
            }
            InterfaceType::Union(index) => {
                let union_types = &types[*index].types;
                let discriminant = next(src).get_u32();
                let ty = union_types.get(discriminant as usize).ok_or_else(|| {
                    anyhow!(
                        "discriminant {} out of range [0..{})",
                        discriminant,
                        union_types.len()
                    )
                })?;
                let value = Rc::new(Self::lift(store, options, &ty, types, src)?);

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            InterfaceType::Option(index) => {
                let discriminant = next(src).get_u32();
                let value = Rc::new(match discriminant {
                    0 => Val::Record(Rc::new([])),
                    1 => Self::lift(store, options, &types[*index], types, src)?,
                    _ => bail!("discriminant {} out of range [0..2)", discriminant),
                });

                Val::Variant {
                    discriminant,
                    value,
                }
            }
            InterfaceType::Expected(index) => {
                let discriminant = next(src).get_u32();
                let value = Rc::new(match discriminant {
                    0 => Self::lift(store, options, &types[*index].ok, types, src)?,
                    1 => Self::lift(store, options, &types[*index].err, types, src)?,
                    _ => bail!("discriminant {} out of range [0..2)", discriminant),
                });

                Val::Variant {
                    discriminant,
                    value,
                }
            }
        })
    }
}
