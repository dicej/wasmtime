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
