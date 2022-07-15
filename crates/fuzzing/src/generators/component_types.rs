//! This module generates test cases for the Wasmtime component model function APIs,
//! e.g. `wasmtime::component::func::Func` and `TypedFunc`.
//!
//! Each case includes a list of arbitrary interface types to use as parameters, plus another one to use as a
//! result, and a component which exports a function and imports a function.  The exported function forwards its
//! parameters to the imported one and forwards the result back to the caller.  This serves to excercise Wasmtime's
//! lifting and lowering code and verify the values remain intact during both processes.

use anyhow::{anyhow, bail, Result};
use arbitrary::{Arbitrary, Unstructured};
use component_test_util::REALLOC_AND_FREE;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::fmt::{self, Debug, Write};
use std::iter;
use std::ops::{ControlFlow, Deref};
use wasmtime::component::{self, Component, Lift, Linker, Lower};
use wasmtime::{Config, Engine, Store, StoreContextMut};
use wasmtime_component_util::{DiscriminantSize, FlagsSize};

const MAX_FLAT_PARAMS: usize = 16;
const MAX_FLAT_RESULTS: usize = 1;
const MAX_ARITY: usize = 5;

/// The name of the imported host function which the generated component will call
pub const IMPORT_FUNCTION: &str = "echo";

/// The name of the exported guest function which the host should call
pub const EXPORT_FUNCTION: &str = "echo";

/// Minimum length of an arbitrary list value generated for a test case
const MIN_LIST_LENGTH: u32 = 0;

/// Maximum length of an arbitrary list value generated for a test case
const MAX_LIST_LENGTH: u32 = 10;

/// Maximum length of an arbitrary tuple type.  As of this writing, the `wasmtime::component::func::typed` module
/// only implements the `ComponentType` trait for tuples up to this length.
const MAX_TUPLE_LENGTH: usize = 16;

#[derive(Copy, Clone, PartialEq, Eq)]
enum CoreType {
    I32,
    I64,
    F32,
    F64,
}

impl CoreType {
    /// This is the `join` operation specified in [the canonical
    /// ABI](https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md#flattening) for
    /// variant types.
    fn join(self, other: Self) -> Self {
        match (self, other) {
            _ if self == other => self,
            (Self::I32, Self::F32) | (Self::F32, Self::I32) => Self::I32,
            _ => Self::I64,
        }
    }
}

impl fmt::Display for CoreType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32 => f.write_str("i32"),
            Self::I64 => f.write_str("i64"),
            Self::F32 => f.write_str("f32"),
            Self::F64 => f.write_str("f64"),
        }
    }
}

/// Wraps a `Box<[T]>` and provides an `Arbitrary` implementation that always generates non-empty slices
#[derive(Debug)]
pub struct NonEmptyArray<T>(Box<[T]>);

impl<'a, T: Arbitrary<'a>> Arbitrary<'a> for NonEmptyArray<T> {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self(
            iter::once(input.arbitrary())
                .chain(input.arbitrary_iter()?)
                .collect::<arbitrary::Result<_>>()?,
        ))
    }
}

/// Wraps a `Box<[T]>` and provides an `Arbitrary` implementation that always generates non-empty slices of length
/// less than or equal to the longest tuple for which Wasmtime generates a `ComponentType` impl
#[derive(Debug)]
pub struct TupleArray<T>(Box<[T]>);

impl<'a, T: Arbitrary<'a>> Arbitrary<'a> for TupleArray<T> {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self(
            iter::once(input.arbitrary())
                .chain(input.arbitrary_iter()?)
                .take(MAX_TUPLE_LENGTH)
                .collect::<arbitrary::Result<_>>()?,
        ))
    }
}

/// Represents a component model interface type
#[allow(missing_docs)]
#[derive(Arbitrary, Debug)]
pub enum Type {
    Unit,
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
    Record(NonEmptyArray<Type>),
    Tuple(TupleArray<Type>),
    Variant(NonEmptyArray<Type>),
    Enum(NonEmptyArray<()>),
    Union(NonEmptyArray<Type>),
    Option(Box<Type>),
    Expected { ok: Box<Type>, err: Box<Type> },
    Flags(NonEmptyArray<()>),
}

impl fmt::Display for Type {
    /// Format this type according to [the component model
    /// grammar](https://github.com/WebAssembly/component-model/blob/main/design/mvp/Explainer.md#type-definitions).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unit => f.write_str("unit"),
            Self::Bool => f.write_str("bool"),
            Self::S8 => f.write_str("s8"),
            Self::U8 => f.write_str("u8"),
            Self::S16 => f.write_str("s16"),
            Self::U16 => f.write_str("u16"),
            Self::S32 => f.write_str("s32"),
            Self::U32 => f.write_str("u32"),
            Self::S64 => f.write_str("s64"),
            Self::U64 => f.write_str("u64"),
            Self::Float32 => f.write_str("float32"),
            Self::Float64 => f.write_str("float64"),
            Self::Char => f.write_str("char"),
            Self::String => f.write_str("string"),
            Self::List(ty) => write!(f, "(list {ty})"),
            Self::Record(types) => {
                f.write_str("(record")?;
                for (index, ty) in types.0.iter().enumerate() {
                    write!(f, r#" (field "f{index}" {ty})"#)?;
                }
                f.write_str(")")
            }
            Self::Tuple(types) => {
                f.write_str("(tuple")?;
                for ty in types.0.iter() {
                    write!(f, r#" {ty}"#)?;
                }
                f.write_str(")")
            }
            Self::Variant(types) => {
                f.write_str("(variant")?;
                for (index, ty) in types.0.iter().enumerate() {
                    write!(f, r#" (case "C{index}" {ty})"#)?;
                }
                f.write_str(")")
            }
            Self::Enum(units) => {
                f.write_str("(enum")?;
                for index in 0..units.0.len() {
                    write!(f, r#" "C{index}""#)?;
                }
                f.write_str(")")
            }
            Self::Union(types) => {
                f.write_str("(union")?;
                for ty in types.0.iter() {
                    write!(f, r#" {ty}"#)?;
                }
                f.write_str(")")
            }
            Self::Option(ty) => {
                write!(f, r#"(option {ty})"#)
            }
            Self::Expected { ok, err } => {
                write!(f, r#"(expected {ok} {err})"#)
            }
            Self::Flags(units) => {
                f.write_str("(flags")?;
                for index in 0..units.0.len() {
                    write!(f, r#" "F{index}""#)?;
                }
                f.write_str(")")
            }
        }
    }
}

fn lower_record<'a>(types: impl Iterator<Item = &'a Type>, vec: &mut Vec<CoreType>) {
    for ty in types {
        ty.lower(vec);
    }
}

fn lower_variant<'a>(types: impl Iterator<Item = &'a Type>, vec: &mut Vec<CoreType>) {
    vec.push(CoreType::I32);
    let offset = vec.len();
    for ty in types {
        for (index, ty) in ty.lowered().iter().enumerate() {
            let index = offset + index;
            if index < vec.len() {
                vec[index] = vec[index].join(*ty);
            } else {
                vec.push(*ty)
            }
        }
    }
}

fn u32_count_from_flag_count(count: usize) -> usize {
    match FlagsSize::from_count(count) {
        FlagsSize::Size1 | FlagsSize::Size2 => 1,
        FlagsSize::Size4Plus(n) => n,
    }
}

struct SizeAndAlignment {
    size: usize,
    alignment: u32,
}

impl Type {
    fn lowered(&self) -> Vec<CoreType> {
        let mut vec = Vec::new();
        self.lower(&mut vec);
        vec
    }

    fn lower(&self, vec: &mut Vec<CoreType>) {
        match self {
            Type::Unit => (),
            Type::Bool
            | Type::U8
            | Type::S8
            | Type::S16
            | Type::U16
            | Type::S32
            | Type::U32
            | Type::Char
            | Type::Enum(_) => vec.push(CoreType::I32),
            Type::S64 | Type::U64 => vec.push(CoreType::I64),
            Type::Float32 => vec.push(CoreType::F32),
            Type::Float64 => vec.push(CoreType::F64),
            Type::String | Type::List(_) => {
                vec.push(CoreType::I32);
                vec.push(CoreType::I32);
            }
            Type::Record(types) => lower_record(types.0.iter(), vec),
            Type::Tuple(types) => lower_record(types.0.iter(), vec),
            Type::Variant(types) | Type::Union(types) => lower_variant(types.0.iter(), vec),
            Type::Option(ty) => lower_variant([&Type::Unit, ty].into_iter(), vec),
            Type::Expected { ok, err } => lower_variant([ok.deref(), err].into_iter(), vec),
            Type::Flags(units) => vec
                .extend(iter::repeat(CoreType::I32).take(u32_count_from_flag_count(units.0.len()))),
        }
    }

    fn size_and_alignment(&self) -> SizeAndAlignment {
        match self {
            Type::Unit => SizeAndAlignment {
                size: 0,
                alignment: 1,
            },

            Type::Bool | Type::S8 | Type::U8 => SizeAndAlignment {
                size: 1,
                alignment: 1,
            },

            Type::S16 | Type::U16 => SizeAndAlignment {
                size: 2,
                alignment: 2,
            },

            Type::S32 | Type::U32 | Type::Char | Type::Float32 => SizeAndAlignment {
                size: 4,
                alignment: 4,
            },

            Type::S64 | Type::U64 | Type::Float64 => SizeAndAlignment {
                size: 8,
                alignment: 8,
            },

            Type::String | Type::List(_) => SizeAndAlignment {
                size: 8,
                alignment: 4,
            },

            Type::Record(types) => record_size_and_alignment(types.0.iter()),

            Type::Tuple(types) => record_size_and_alignment(types.0.iter()),

            Type::Variant(types) | Type::Union(types) => variant_size_and_alignment(types.0.iter()),

            Type::Enum(units) => variant_size_and_alignment(units.0.iter().map(|_| &Type::Unit)),

            Type::Option(ty) => variant_size_and_alignment([&Type::Unit, ty].into_iter()),

            Type::Expected { ok, err } => variant_size_and_alignment([ok.deref(), err].into_iter()),

            Type::Flags(units) => match FlagsSize::from_count(units.0.len()) {
                FlagsSize::Size1 => SizeAndAlignment {
                    size: 1,
                    alignment: 1,
                },
                FlagsSize::Size2 => SizeAndAlignment {
                    size: 2,
                    alignment: 2,
                },
                FlagsSize::Size4Plus(n) => SizeAndAlignment {
                    size: n * 4,
                    alignment: 4,
                },
            },
        }
    }
}

fn align_to(a: usize, align: u32) -> usize {
    let align = align as usize;
    (a + (align - 1)) & !(align - 1)
}

fn record_size_and_alignment<'a>(types: impl Iterator<Item = &'a Type>) -> SizeAndAlignment {
    let mut offset = 0;
    let mut align = 1;
    for ty in types {
        let SizeAndAlignment { size, alignment } = ty.size_and_alignment();
        offset = align_to(offset, alignment) + size;
        align = align.max(alignment);
    }

    SizeAndAlignment {
        size: align_to(offset, align),
        alignment: align,
    }
}

fn variant_size_and_alignment<'a>(
    types: impl ExactSizeIterator<Item = &'a Type>,
) -> SizeAndAlignment {
    let discriminant_size = DiscriminantSize::from_count(types.len()).unwrap();
    let mut alignment = u32::from(discriminant_size);
    let mut size = 0;
    for ty in types {
        let size_and_alignment = ty.size_and_alignment();
        alignment = alignment.max(size_and_alignment.alignment);
        size = size.max(size_and_alignment.size);
    }

    SizeAndAlignment {
        size: align_to(usize::from(discriminant_size), alignment) + size,
        alignment,
    }
}

/// Represents an instance of a component interface type
#[allow(missing_docs)]
#[derive(Debug)]
pub enum Value {
    Bool(bool),
    S8(i8),
    U8(u8),
    S16(i16),
    U16(u16),
    S32(i32),
    U32(u32),
    S64(i64),
    U64(u64),
    Float32(f32),
    Float64(f64),
    Char(char),
    String(Box<str>),
    List(Box<[Value]>),
    Record(Box<[Value]>),
    Variant {
        discriminant: usize,
        value: Box<Value>,
    },
    Flags {
        count: usize,
        value: Box<[u32]>,
    },
}

impl Value {
    /// Generate an arbitrary instance of the specified type.
    pub fn arbitrary(ty: &Type, input: &mut Unstructured) -> arbitrary::Result<Self> {
        Ok(match &ty {
            Type::Unit => Value::Record(Box::new([])),
            Type::Bool => Value::Bool(input.arbitrary()?),
            Type::S8 => Value::S8(input.arbitrary()?),
            Type::U8 => Value::U8(input.arbitrary()?),
            Type::S16 => Value::S16(input.arbitrary()?),
            Type::U16 => Value::U16(input.arbitrary()?),
            Type::S32 => Value::S32(input.arbitrary()?),
            Type::U32 => Value::U32(input.arbitrary()?),
            Type::S64 => Value::S64(input.arbitrary()?),
            Type::U64 => Value::U64(input.arbitrary()?),
            Type::Float32 => Value::Float32(input.arbitrary()?),
            Type::Float64 => Value::Float64(input.arbitrary()?),
            Type::Char => Value::Char(input.arbitrary()?),
            Type::String => Value::String(input.arbitrary()?),
            Type::List(ty) => {
                let mut values = Vec::new();
                input.arbitrary_loop(Some(MIN_LIST_LENGTH), Some(MAX_LIST_LENGTH), |input| {
                    values.push(Value::arbitrary(ty, input)?);

                    Ok(ControlFlow::Continue(()))
                })?;
                Value::List(values.into())
            }
            Type::Record(types) => Value::Record(
                types
                    .0
                    .iter()
                    .map(|ty| Value::arbitrary(ty, input))
                    .collect::<arbitrary::Result<_>>()?,
            ),
            Type::Tuple(types) => Value::Record(
                types
                    .0
                    .iter()
                    .map(|ty| Value::arbitrary(ty, input))
                    .collect::<arbitrary::Result<_>>()?,
            ),
            Type::Variant(types) => {
                let discriminant = input.int_in_range(0..=types.0.len() - 1)?;
                Value::Variant {
                    discriminant,
                    value: Box::new(Value::arbitrary(&types.0[discriminant], input)?),
                }
            }
            Type::Enum(units) => {
                let discriminant = input.int_in_range(0..=units.0.len() - 1)?;
                Value::Variant {
                    discriminant,
                    value: Box::new(Value::Record(Box::new([]))),
                }
            }
            Type::Union(types) => {
                let discriminant = input.int_in_range(0..=types.0.len() - 1)?;
                Value::Variant {
                    discriminant,
                    value: Box::new(Value::arbitrary(&types.0[discriminant], input)?),
                }
            }
            Type::Option(ty) => {
                let discriminant = input.int_in_range(0..=1)?;
                Value::Variant {
                    discriminant,
                    value: if discriminant == 0 {
                        Box::new(Value::Record(Box::new([])))
                    } else {
                        Box::new(Value::arbitrary(ty, input)?)
                    },
                }
            }
            Type::Expected { ok, err } => {
                let discriminant = input.int_in_range(0..=1)?;
                Value::Variant {
                    discriminant,
                    value: if discriminant == 0 {
                        Box::new(Value::arbitrary(ok, input)?)
                    } else {
                        Box::new(Value::arbitrary(err, input)?)
                    },
                }
            }
            Type::Flags(units) => Value::Flags {
                count: units.0.len(),
                value: iter::repeat_with(|| input.arbitrary())
                    .take(u32_count_from_flag_count(units.0.len()))
                    .collect::<arbitrary::Result<_>>()?,
            },
        })
    }

    /// Attempt to convert this value to a [`component::Val`] of the specified type.
    pub fn to_val(&self, ty: &component::Type) -> anyhow::Result<component::Val> {
        Ok(match (self, ty) {
            (Value::Record(values), component::Type::Unit) if values.is_empty() => {
                component::Val::Unit
            }
            (Value::Bool(value), component::Type::Bool) => component::Val::Bool(*value),
            (Value::S8(value), component::Type::S8) => component::Val::S8(*value),
            (Value::U8(value), component::Type::U8) => component::Val::U8(*value),
            (Value::S16(value), component::Type::S16) => component::Val::S16(*value),
            (Value::U16(value), component::Type::U16) => component::Val::U16(*value),
            (Value::S32(value), component::Type::S32) => component::Val::S32(*value),
            (Value::U32(value), component::Type::U32) => component::Val::U32(*value),
            (Value::S64(value), component::Type::S64) => component::Val::S64(*value),
            (Value::U64(value), component::Type::U64) => component::Val::U64(*value),
            (Value::Float32(value), component::Type::Float32) => {
                component::Val::Float32(value.to_bits())
            }
            (Value::Float64(value), component::Type::Float64) => {
                component::Val::Float64(value.to_bits())
            }
            (Value::Char(value), component::Type::Char) => component::Val::Char(*value),
            (Value::String(value), component::Type::String) => {
                component::Val::String(value.clone())
            }
            (Value::List(values), component::Type::List(list)) => list.new_val(
                values
                    .iter()
                    .map(|v| v.to_val(&list.ty()))
                    .collect::<Result<_>>()?,
            )?,
            (Value::Record(values), component::Type::Record(record)) => record.new_val(
                values
                    .iter()
                    .zip(record.fields())
                    .map(|(v, field)| Ok((field.name, v.to_val(&field.ty)?)))
                    .collect::<Result<Vec<_>>>()?,
            )?,
            (Value::Record(values), component::Type::Tuple(tuple)) => tuple.new_val(
                values
                    .iter()
                    .zip(tuple.types())
                    .map(|(v, ty)| v.to_val(&ty))
                    .collect::<Result<_>>()?,
            )?,
            (
                Value::Variant {
                    discriminant,
                    value,
                },
                component::Type::Variant(variant),
            ) => variant.new_val(
                &format!("C{discriminant}"),
                value.to_val(
                    &variant
                        .cases()
                        .nth(*discriminant)
                        .ok_or_else(|| anyhow!("discriminant out of range"))?
                        .ty,
                )?,
            )?,
            (
                Value::Variant {
                    discriminant,
                    value,
                },
                component::Type::Enum(en),
            ) if value.to_val(&component::Type::Unit).is_ok() => {
                en.new_val(&format!("C{discriminant}"))?
            }
            (
                Value::Variant {
                    discriminant,
                    value,
                },
                component::Type::Union(un),
            ) => un.new_val(
                u32::try_from(*discriminant)?,
                value.to_val(
                    &un.types()
                        .nth(*discriminant)
                        .ok_or_else(|| anyhow!("discriminant out of range"))?,
                )?,
            )?,
            (
                Value::Variant {
                    discriminant,
                    value,
                },
                component::Type::Option(option),
            ) => option.new_val(match discriminant {
                0 => None,
                1 => Some(value.to_val(&option.ty())?),
                _ => bail!("discriminant out of range"),
            })?,
            (
                Value::Variant {
                    discriminant,
                    value,
                },
                component::Type::Expected(expected),
            ) => expected.new_val(match discriminant {
                0 => Ok(value.to_val(&expected.ok())?),
                1 => Err(value.to_val(&expected.err())?),
                _ => bail!("discriminant out of range"),
            })?,
            (Value::Flags { count, value }, component::Type::Flags(flags)) => flags.new_val(
                &(0..*count)
                    .zip(flags.names())
                    .filter_map(|(index, name)| {
                        if value[index / 32] & (1 << (index % 32)) != 0 {
                            Some(name)
                        } else {
                            None
                        }
                    })
                    .collect::<Box<_>>(),
            )?,
            _ => bail!("type mismatch: {self:?} vs. {ty:?}"),
        })
    }
}

fn make_import_and_export(params: &[Type], result: &Type) -> Box<str> {
    let params_lowered = params
        .iter()
        .flat_map(|ty| ty.lowered())
        .collect::<Box<[_]>>();
    let result_lowered = result.lowered();

    let mut core_params = String::new();
    let mut gets = String::new();

    if params_lowered.len() <= MAX_FLAT_PARAMS {
        for (index, param) in params_lowered.iter().enumerate() {
            write!(&mut core_params, " {param}").unwrap();
            write!(&mut gets, "local.get {index} ").unwrap();
        }
    } else {
        write!(&mut core_params, " i32").unwrap();
        write!(&mut gets, "local.get 0 ").unwrap();
    }

    let maybe_core_params = if params_lowered.is_empty() {
        String::new()
    } else {
        format!("(param{core_params})")
    };

    if result_lowered.len() <= MAX_FLAT_RESULTS {
        let mut core_results = String::new();
        for result in result_lowered.iter() {
            write!(&mut core_results, " {result}").unwrap();
        }

        let maybe_core_results = if result_lowered.is_empty() {
            String::new()
        } else {
            format!("(result{core_results})")
        };

        format!(
            r#"
            (func $f (import "host" "{IMPORT_FUNCTION}") {maybe_core_params} {maybe_core_results})

            (func (export "{EXPORT_FUNCTION}") {maybe_core_params} {maybe_core_results}
                {gets}

                call $f
            )"#
        )
    } else {
        let SizeAndAlignment { size, alignment } = result.size_and_alignment();

        format!(
            r#"
            (func $f (import "host" "{IMPORT_FUNCTION}") (param{core_params} i32))

            (func (export "{EXPORT_FUNCTION}") {maybe_core_params} (result i32)
                (local $base i32)
                (local.set $base
                    (call $realloc
                        (i32.const 0)
                        (i32.const 0)
                        (i32.const {alignment})
                        (i32.const {size})))
                {gets}
                local.get $base

                call $f

                local.get $base
            )"#
        )
    }
    .into()
}

fn make_name<'a>(name_counter: &mut u32) -> Ident {
    let name = format_ident!("Foo{name_counter}");
    *name_counter += 1;
    name
}

/// Generate a [`TokenStream`] containing the rust type name for a type.
///
/// The `name_counter` parameter is used to generate names for each recursively visited type.  The `declarations`
/// parameter is used to accumulate declarations for each recursively visited type.
pub fn rust_type<'a>(
    ty: &'a Type,
    name_counter: &mut u32,
    declarations: &mut TokenStream,
) -> TokenStream {
    match ty {
        Type::Unit => quote!(()),
        Type::Bool => quote!(bool),
        Type::S8 => quote!(i8),
        Type::U8 => quote!(u8),
        Type::S16 => quote!(i16),
        Type::U16 => quote!(u16),
        Type::S32 => quote!(i32),
        Type::U32 => quote!(u32),
        Type::S64 => quote!(i64),
        Type::U64 => quote!(u64),
        Type::Float32 => quote!(Float32),
        Type::Float64 => quote!(Float64),
        Type::Char => quote!(char),
        Type::String => quote!(Box<str>),
        Type::List(ty) => {
            let ty = rust_type(ty, name_counter, declarations);
            quote!(Vec<#ty>)
        }
        Type::Record(types) => {
            let fields = types
                .0
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    let name = format_ident!("f{index}");
                    let ty = rust_type(ty, name_counter, declarations);
                    quote!(#name: #ty,)
                })
                .collect::<TokenStream>();

            let name = make_name(name_counter);

            declarations.extend(quote! {
                #[derive(ComponentType, Lift, Lower, PartialEq, Debug, Clone, Arbitrary)]
                #[component(record)]
                struct #name {
                    #fields
                }
            });

            quote!(#name)
        }
        Type::Tuple(types) => {
            let fields = types
                .0
                .iter()
                .map(|ty| {
                    let ty = rust_type(ty, name_counter, declarations);
                    quote!(#ty,)
                })
                .collect::<TokenStream>();

            quote!((#fields))
        }
        Type::Variant(types) | Type::Union(types) => {
            let cases = types
                .0
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    let name = format_ident!("C{index}");
                    let ty = rust_type(ty, name_counter, declarations);
                    quote!(#name(#ty),)
                })
                .collect::<TokenStream>();

            let name = make_name(name_counter);

            let which = if let Type::Variant(_) = ty {
                quote!(variant)
            } else {
                quote!(union)
            };

            declarations.extend(quote! {
                #[derive(ComponentType, Lift, Lower, PartialEq, Debug, Clone, Arbitrary)]
                #[component(#which)]
                enum #name {
                    #cases
                }
            });

            quote!(#name)
        }
        Type::Enum(units) => {
            let cases = (0..units.0.len())
                .map(|index| {
                    let name = format_ident!("C{index}");
                    quote!(#name,)
                })
                .collect::<TokenStream>();

            let name = make_name(name_counter);

            declarations.extend(quote! {
                #[derive(ComponentType, Lift, Lower, PartialEq, Debug, Clone, Arbitrary)]
                #[component(enum)]
                enum #name {
                    #cases
                }
            });

            quote!(#name)
        }
        Type::Option(ty) => {
            let ty = rust_type(ty, name_counter, declarations);
            quote!(Option<#ty>)
        }
        Type::Expected { ok, err } => {
            let ok = rust_type(ok, name_counter, declarations);
            let err = rust_type(err, name_counter, declarations);
            quote!(Result<#ok, #err>)
        }
        Type::Flags(units) => {
            let type_name = make_name(name_counter);

            let mut flags = TokenStream::new();
            let mut names = TokenStream::new();

            for index in 0..units.0.len() {
                let name = format_ident!("F{index}");
                flags.extend(quote!(const #name;));
                names.extend(quote!(#type_name::#name,))
            }

            declarations.extend(quote! {
                wasmtime::component::flags! {
                    #type_name {
                        #flags
                    }
                }

                impl<'a> Arbitrary<'a> for #type_name {
                    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
                        let mut flags = #type_name::default();
                        for flag in [#names] {
                            if input.arbitrary()? {
                                flags |= flag;
                            }
                        }
                        Ok(flags)
                    }
                }
            });

            quote!(#type_name)
        }
    }
}

/// Represents a test case for calling a component function
#[derive(Debug)]
pub struct TestCase {
    /// The types of parameters to pass to the function
    pub params: Box<[Type]>,
    /// The type of the result to be returned by the function
    pub result: Type,
    /// A WAT fragment representing the core function import and export to use for testing
    pub import_and_export: Box<str>,
}

impl<'a> Arbitrary<'a> for TestCase {
    /// Generate an arbitrary [`TestCase`].
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        crate::init_fuzzing();

        let params = input
            .arbitrary_iter()?
            .take(MAX_ARITY)
            .collect::<arbitrary::Result<Box<[_]>>>()?;
        let result = input.arbitrary()?;
        let import_and_export = make_import_and_export(&params, &result);

        Ok(Self {
            params,
            result,
            import_and_export,
        })
    }
}

/// Generate a complete WAT file based on the specified fragments.
///
/// `params` should contain the "lifted" component model parameters, if any, while the `result` should contain the
/// "lifted" result, if any.
///
/// `import_and_export` should be a [`TestCase::import_and_export`].
pub fn make_component(params: &str, result: &str, import_and_export: &str) -> Box<str> {
    format!(
        r#"
        (component
            (core module $libc
                (memory (export "memory") 1)
                {REALLOC_AND_FREE}
            )

            (core instance $libc (instantiate $libc))

            (import "{IMPORT_FUNCTION}" (func $f {params} {result}))

            (core func $f_lower (canon lower (func $f) (memory $libc "memory") (realloc (func $libc "realloc"))))

            (core module $m
                (memory (import "libc" "memory") 1)
                (func $realloc (import "libc" "realloc") (param i32 i32 i32 i32) (result i32))

                {import_and_export}
            )

            (core instance $i (instantiate $m
                (with "libc" (instance $libc))
                (with "host" (instance (export "{IMPORT_FUNCTION}" (func $f_lower))))
            ))

            (func (export "echo") {params} {result}
                (canon lift
                    (core func $i "echo")
                    (memory $libc "memory")
                    (realloc (func $libc "realloc"))
                )
            )
        )"#,
    )
    .into()
}

macro_rules! define_static_api_test {
    ($name:ident $(($param:ident $param_name:ident $param_expected_name:ident))*) => {
        #[allow(unused_parens)]
        /// Generate zero or more sets of arbitrary argument and result values and execute the test using those
        /// values, asserting that they flow from host-to-guest and guest-to-host unchanged.
        pub fn $name<'a, $($param,)* R>(
            input: &mut Unstructured<'a>,
            params: &str, result: &str,
            import_and_export: &str
        ) -> arbitrary::Result<()>
        where
            $($param: Lift + Lower + Clone + PartialEq + Debug + Arbitrary<'a> + 'static,)*
            R: Lift + Lower + Clone + PartialEq + Debug + Arbitrary<'a> + 'static
        {
            let mut config = Config::new();
            config.wasm_component_model(true);
            let engine = Engine::new(&config).unwrap();
            let component = Component::new(
                &engine,
                make_component(params, result, import_and_export).as_bytes()
            ).unwrap();
            let mut linker = Linker::new(&engine);
            linker
                .root()
                .func_wrap(
                    IMPORT_FUNCTION,
                    |cx: StoreContextMut<'_, ($(Option<$param>,)* Option<R>)>,
                    $($param_name: $param,)*|
                      -> Result<R>
                    {
                        let ($($param_expected_name,)* result) = cx.data();
                        $(assert_eq!($param_name, *$param_expected_name.as_ref().unwrap());)*
                        Ok(result.as_ref().unwrap().clone())
                    },
                )
                .unwrap();
            let mut store = Store::new(&engine, Default::default());
            let instance = linker.instantiate(&mut store, &component).unwrap();
            let func = instance
                .get_typed_func::<($($param,)*), R, _>(&mut store, EXPORT_FUNCTION)
                .unwrap();

            while input.arbitrary()? {
                $(let $param_name = input.arbitrary::<$param>()?;)*
                let result = input.arbitrary::<R>()?;
                *store.data_mut() = ($(Some($param_name.clone()),)* Some(result.clone()));

                assert_eq!(func.call(&mut store, ($($param_name,)*)).unwrap(), result);
                func.post_return(&mut store).unwrap();
            }

            Ok(())
        }
    }
}

define_static_api_test!(static_api_test0);
define_static_api_test!(static_api_test1 (P0 p0 p0_expected));
define_static_api_test!(static_api_test2 (P0 p0 p0_expected) (P1 p1 p1_expected));
define_static_api_test!(static_api_test3 (P0 p0 p0_expected) (P1 p1 p1_expected) (P2 p2 p2_expected));
define_static_api_test!(static_api_test4 (P0 p0 p0_expected) (P1 p1 p1_expected) (P2 p2 p2_expected)
                        (P3 p3 p3_expected));
define_static_api_test!(static_api_test5 (P0 p0 p0_expected) (P1 p1 p1_expected) (P2 p2 p2_expected)
                        (P3 p3 p3_expected) (P4 p4 p4_expected));
