//! This module generates test cases for the Wasmtime component model function APIs,
//! e.g. `wasmtime::component::func::Func` and `TypedFunc`.
//!
//! Each case includes a list of arbitrary interface types to use as parameters, plus another one to use as a
//! result, and a component which exports a function and imports a function.  The exported function forwards its
//! parameters to the imported one and forwards the result back to the caller.  This serves to excercise Wasmtime's
//! lifting and lowering code and verify the values remain intact during both processes.

use anyhow::Result;
use arbitrary::{Arbitrary, Unstructured};
use component_test_util::REALLOC_AND_FREE;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::fmt::{self, Debug, Write};
use std::iter;
use std::ops::{ControlFlow, Deref};
use wasmtime::component::{self, Component, Lift, Linker, Lower, Val};
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

/// Generate an arbitrary instance of the specified type.
pub fn arbitrary_val(ty: &component::Type, input: &mut Unstructured) -> arbitrary::Result<Val> {
    use component::Type;

    Ok(match ty {
        Type::Unit => Val::Unit,
        Type::Bool => Val::Bool(input.arbitrary()?),
        Type::S8 => Val::S8(input.arbitrary()?),
        Type::U8 => Val::U8(input.arbitrary()?),
        Type::S16 => Val::S16(input.arbitrary()?),
        Type::U16 => Val::U16(input.arbitrary()?),
        Type::S32 => Val::S32(input.arbitrary()?),
        Type::U32 => Val::U32(input.arbitrary()?),
        Type::S64 => Val::S64(input.arbitrary()?),
        Type::U64 => Val::U64(input.arbitrary()?),
        Type::Float32 => Val::Float32(input.arbitrary::<f32>()?.to_bits()),
        Type::Float64 => Val::Float64(input.arbitrary::<f64>()?.to_bits()),
        Type::Char => Val::Char(input.arbitrary()?),
        Type::String => Val::String(input.arbitrary()?),
        Type::List(list) => {
            let mut values = Vec::new();
            input.arbitrary_loop(Some(MIN_LIST_LENGTH), Some(MAX_LIST_LENGTH), |input| {
                values.push(arbitrary_val(&list.ty(), input)?);

                Ok(ControlFlow::Continue(()))
            })?;

            list.new_val(values.into()).unwrap()
        }
        Type::Record(record) => record
            .new_val(
                record
                    .fields()
                    .map(|field| Ok((field.name, arbitrary_val(&field.ty, input)?)))
                    .collect::<arbitrary::Result<Vec<_>>>()?,
            )
            .unwrap(),
        Type::Tuple(tuple) => tuple
            .new_val(
                tuple
                    .types()
                    .map(|ty| arbitrary_val(&ty, input))
                    .collect::<arbitrary::Result<_>>()?,
            )
            .unwrap(),
        Type::Variant(variant) => {
            let mut cases = variant.cases();
            let discriminant = input.int_in_range(0..=cases.len() - 1)?;
            variant
                .new_val(
                    &format!("C{discriminant}"),
                    arbitrary_val(&cases.nth(discriminant).unwrap().ty, input)?,
                )
                .unwrap()
        }
        Type::Enum(en) => {
            let discriminant = input.int_in_range(0..=en.names().len() - 1)?;
            en.new_val(&format!("C{discriminant}")).unwrap()
        }
        Type::Union(un) => {
            let mut types = un.types();
            let discriminant = input.int_in_range(0..=types.len() - 1)?;
            un.new_val(
                discriminant.try_into().unwrap(),
                arbitrary_val(&types.nth(discriminant).unwrap(), input)?,
            )
            .unwrap()
        }
        Type::Option(option) => {
            let discriminant = input.int_in_range(0..=1)?;
            option
                .new_val(match discriminant {
                    0 => None,
                    1 => Some(arbitrary_val(&option.ty(), input)?),
                    _ => unreachable!(),
                })
                .unwrap()
        }
        Type::Expected(expected) => {
            let discriminant = input.int_in_range(0..=1)?;
            expected
                .new_val(match discriminant {
                    0 => Ok(arbitrary_val(&expected.ok(), input)?),
                    1 => Err(arbitrary_val(&expected.err(), input)?),
                    _ => unreachable!(),
                })
                .unwrap()
        }
        Type::Flags(flags) => flags
            .new_val(
                &flags
                    .names()
                    .filter_map(|name| {
                        input
                            .arbitrary()
                            .map(|p| if p { Some(name) } else { None })
                            .transpose()
                    })
                    .collect::<arbitrary::Result<Box<[_]>>>()?,
            )
            .unwrap(),
    })
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

fn make_rust_name(name_counter: &mut u32) -> Ident {
    let name = format_ident!("Foo{name_counter}");
    *name_counter += 1;
    name
}

/// Generate a [`TokenStream`] containing the rust type name for a type.
///
/// The `name_counter` parameter is used to generate names for each recursively visited type.  The `declarations`
/// parameter is used to accumulate declarations for each recursively visited type.
pub fn rust_type(ty: &Type, name_counter: &mut u32, declarations: &mut TokenStream) -> TokenStream {
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

            let name = make_rust_name(name_counter);

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

            let name = make_rust_name(name_counter);

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

            let name = make_rust_name(name_counter);

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
            let type_name = make_rust_name(name_counter);

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

fn make_component_name(name_counter: &mut u32) -> String {
    let name = format!("$Foo{name_counter}");
    *name_counter += 1;
    name
}

fn write_component_type(
    ty: &Type,
    f: &mut String,
    name_counter: &mut u32,
    declarations: &mut String,
) {
    match ty {
        Type::Unit => f.push_str("unit"),
        Type::Bool => f.push_str("bool"),
        Type::S8 => f.push_str("s8"),
        Type::U8 => f.push_str("u8"),
        Type::S16 => f.push_str("s16"),
        Type::U16 => f.push_str("u16"),
        Type::S32 => f.push_str("s32"),
        Type::U32 => f.push_str("u32"),
        Type::S64 => f.push_str("s64"),
        Type::U64 => f.push_str("u64"),
        Type::Float32 => f.push_str("float32"),
        Type::Float64 => f.push_str("float64"),
        Type::Char => f.push_str("char"),
        Type::String => f.push_str("string"),
        Type::List(ty) => {
            let mut case = String::new();
            write_component_type(ty, &mut case, name_counter, declarations);
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (list {case}))").unwrap();
            f.push_str(&name);
        }
        Type::Record(types) => {
            let mut fields = String::new();
            for (index, ty) in types.0.iter().enumerate() {
                write!(fields, r#" (field "f{index}" "#).unwrap();
                write_component_type(ty, &mut fields, name_counter, declarations);
                fields.push_str(")");
            }
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (record{fields}))").unwrap();
            f.push_str(&name);
        }
        Type::Tuple(types) => {
            let mut fields = String::new();
            for ty in types.0.iter() {
                fields.push_str(" ");
                write_component_type(ty, &mut fields, name_counter, declarations);
            }
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (tuple{fields}))").unwrap();
            f.push_str(&name);
        }
        Type::Variant(types) => {
            let mut cases = String::new();
            for (index, ty) in types.0.iter().enumerate() {
                write!(cases, r#" (case "C{index}" "#).unwrap();
                write_component_type(ty, &mut cases, name_counter, declarations);
                cases.push_str(")");
            }
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (variant{cases}))").unwrap();
            f.push_str(&name);
        }
        Type::Enum(units) => {
            f.push_str("(enum");
            for index in 0..units.0.len() {
                write!(f, r#" "C{index}""#).unwrap();
            }
            f.push_str(")");
        }
        Type::Union(types) => {
            let mut cases = String::new();
            for ty in types.0.iter() {
                cases.push_str(" ");
                write_component_type(ty, &mut cases, name_counter, declarations);
            }
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (union{cases}))").unwrap();
            f.push_str(&name);
        }
        Type::Option(ty) => {
            let mut case = String::new();
            write_component_type(ty, &mut case, name_counter, declarations);
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (option {case}))").unwrap();
            f.push_str(&name);
        }
        Type::Expected { ok, err } => {
            let mut cases = String::new();
            write_component_type(ok, &mut cases, name_counter, declarations);
            cases.push_str(" ");
            write_component_type(err, &mut cases, name_counter, declarations);
            let name = make_component_name(name_counter);
            write!(declarations, "(type {name} (expected {cases}))").unwrap();
            f.push_str(&name);
        }
        Type::Flags(units) => {
            f.push_str("(flags");
            for index in 0..units.0.len() {
                write!(f, r#" "F{index}""#).unwrap();
            }
            f.push_str(")");
        }
    }
}

/// Represents custom fragments of a WAT file which may be used to create a component for exercising [`TestCase`]s
#[derive(Debug)]
pub struct Declarations {
    /// Type declarations (if any) referenced by `params` and/or `result`
    pub types: Box<str>,
    /// Parameter declarations used for the imported and exported functions
    pub params: Box<str>,
    /// Result declaration used for the imported and exported functions
    pub result: Box<str>,
    /// A WAT fragment representing the core function import and export to use for testing
    pub import_and_export: Box<str>,
}

impl Declarations {
    /// Generate a complete WAT file based on the specified fragments.
    pub fn make_component(&self) -> Box<str> {
        let Self {
            types,
            params,
            result,
            import_and_export,
        } = self;

        format!(
            r#"
            (component
                (core module $libc
                    (memory (export "memory") 1)
                    {REALLOC_AND_FREE}
                )

                (core instance $libc (instantiate $libc))

                {types}

                (import "{IMPORT_FUNCTION}" (func $f {params} {result}))

                (core func $f_lower (canon lower
                    (func $f)
                    (memory $libc "memory")
                    (realloc (func $libc "realloc"))
                ))

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
}

/// Represents a test case for calling a component function
#[derive(Debug)]
pub struct TestCase {
    /// The types of parameters to pass to the function
    pub params: Box<[Type]>,
    /// The type of the result to be returned by the function
    pub result: Type,
}

impl TestCase {
    /// Generate a `Declarations` for this `TestCase` which may be used to build a component to execute the case.
    pub fn declarations(&self) -> Declarations {
        let mut types = String::new();
        let name_counter = &mut 0;

        let params = self
            .params
            .iter()
            .map(|ty| {
                let mut tmp = String::new();
                write_component_type(ty, &mut tmp, name_counter, &mut types);
                format!("(param {tmp})")
            })
            .collect::<Box<[_]>>()
            .join(" ")
            .into();

        let result = {
            let mut tmp = String::new();
            write_component_type(&self.result, &mut tmp, name_counter, &mut types);
            format!("(result {tmp})")
        }
        .into();

        let import_and_export = make_import_and_export(&self.params, &self.result);

        Declarations {
            types: types.into(),
            params,
            result,
            import_and_export,
        }
    }
}

impl<'a> Arbitrary<'a> for TestCase {
    /// Generate an arbitrary [`TestCase`].
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        crate::init_fuzzing();

        Ok(Self {
            params: input
                .arbitrary_iter()?
                .take(MAX_ARITY)
                .collect::<arbitrary::Result<Box<[_]>>>()?,
            result: input.arbitrary()?,
        })
    }
}

macro_rules! define_static_api_test {
    ($name:ident $(($param:ident $param_name:ident $param_expected_name:ident))*) => {
        #[allow(unused_parens)]
        /// Generate zero or more sets of arbitrary argument and result values and execute the test using those
        /// values, asserting that they flow from host-to-guest and guest-to-host unchanged.
        pub fn $name<'a, $($param,)* R>(
            input: &mut Unstructured<'a>,
            declarations: &Declarations,
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
                declarations.make_component().as_bytes()
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
