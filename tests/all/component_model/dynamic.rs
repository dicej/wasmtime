use super::{make_echo_component, make_echo_component_with_params, Type};
use anyhow::Result;
use std::rc::Rc;
use wasmtime::component::{Component, Linker, Val};
use wasmtime::Store;

#[test]
fn primitives() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    for (input, ty, param) in [
        (Val::Bool(true), "bool", Type::U8),
        (Val::S8(-42), "s8", Type::S8),
        (Val::U8(42), "u8", Type::U8),
        (Val::S16(-4242), "s16", Type::S16),
        (Val::U16(4242), "u16", Type::U16),
        (Val::S32(-314159265), "s32", Type::I32),
        (Val::U32(314159265), "u32", Type::I32),
        (Val::S64(-31415926535897), "s64", Type::I64),
        (Val::U64(31415926535897), "u64", Type::I64),
        (Val::Float32(3.14159265_f32.to_bits()), "float32", Type::F32),
        (Val::Float64(3.14159265_f64.to_bits()), "float64", Type::F64),
        (Val::Char('🦀'), "char", Type::I32),
    ] {
        let component = Component::new(&engine, make_echo_component_with_params(ty, &[param]))?;
        let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

        let output = instance
            .get_func(&mut store, "echo")
            .unwrap()
            .call(&mut store, &[input.clone()])?;

        assert_eq!(input, output);
    }

    Ok(())
}

#[test]
fn strings() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("string", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::String(Rc::from("hello, component!"));
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}

#[test]
fn lists() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::List(Rc::new([
        Val::U32(32343),
        Val::U32(79023439),
        Val::U32(2084037802),
    ]));
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}

#[test]
fn records() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"(record (field "A" u32) (field "B" float64) (field "C" (record (field "D" bool) (field "E" u32))))"#,
            &[Type::I32, Type::F64, Type::U8, Type::I32],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::Record(Rc::new([
        Val::U32(32343),
        Val::Float64(3.14159265_f64.to_bits()),
        Val::Record(Rc::new([Val::Bool(false), Val::U32(2084037802)])),
    ]));
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}

#[test]
fn variants() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"(variant (case "A" u32) (case "B" float64) (case "C" (record (field "D" bool) (field "E" u32))))"#,
            &[Type::U8, Type::I64, Type::I32],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::Variant {
        discriminant: 1,
        value: Rc::new(Val::Float64(3.14159265_f64.to_bits())),
    };
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}

#[test]
fn flags() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(r#"(flags "A" "B" "C" "D" "E")"#, &[Type::U8]),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::Flags {
        count: 5,
        value: Rc::new([0b10101]),
    };
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}

#[test]
fn everything() -> Result<()> {
    // This serves to test both nested types and storing parameters on the heap (i.e. exceeding `MAX_STACK_PARAMS)

    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(
        &engine,
        make_echo_component_with_params(
            r#"
            (record
                (field "A" u32)
                (field "B" (enum "1" "2"))
                (field "C" (record (field "D" bool) (field "E" u32)))
                (field "F" (list (flags "G" "H" "I")))
                (field "J" (variant
                               (case "K" u32)
                               (case "L" float64)
                               (case "M" (record (field "N" bool) (field "O" u32)))))
                (field "P" s8)
                (field "Q" s16)
                (field "R" s32)
                (field "S" s64)
                (field "T" float32)
                (field "U" float64)
                (field "V" string)
                (field "W" char)
                (field "X" unit)
                (field "Y" (tuple u32 u32))
                (field "Z" (union u32 float64))
            )"#,
            &[
                Type::I32,
                Type::U8,
                Type::U8,
                Type::I32,
                Type::I32,
                Type::I32,
                Type::U8,
                Type::I64,
                Type::I32,
                Type::S8,
                Type::S16,
                Type::I32,
                Type::I64,
                Type::F32,
                Type::F64,
                Type::I32,
                Type::I32,
                Type::I32,
                Type::I32,
                Type::I32,
                Type::I64,
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::Record(Rc::new([
        Val::U32(32343),
        Val::Variant {
            discriminant: 1,
            value: Rc::new(Val::Record(Rc::new([]))),
        },
        Val::Record(Rc::new([Val::Bool(false), Val::U32(2084037802)])),
        Val::List(Rc::new([
            Val::Flags {
                count: 3,
                value: Rc::new([0b101]),
            },
            Val::Flags {
                count: 3,
                value: Rc::new([0b010]),
            },
        ])),
        Val::Variant {
            discriminant: 1,
            value: Rc::new(Val::Float64(3.14159265_f64.to_bits())),
        },
        Val::S8(42),
        Val::S16(4242),
        Val::S32(42424242),
        Val::S64(424242424242424242),
        Val::Float32(3.14159265_f32.to_bits()),
        Val::Float64(3.14159265_f64.to_bits()),
        Val::String(Rc::from("wow, nice types")),
        Val::Char('🦀'),
        Val::Record(Rc::new([])),
        Val::Record(Rc::new([Val::U32(42), Val::U32(24)])),
        Val::Variant {
            discriminant: 1,
            value: Rc::new(Val::Float64(3.14159265_f64.to_bits())),
        },
    ]));
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}
