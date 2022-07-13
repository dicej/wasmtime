use super::{make_echo_component_with_primitive, Primitive};
use anyhow::Result;
use wasmtime::component::{Component, Linker, Val};
use wasmtime::Store;

#[test]
fn primitives() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    for (input, ty, size, primitive) in [
        (Val::Bool(true), "bool", 1, Primitive::I32),
        (Val::S8(-42), "s8", 1, Primitive::I32),
        (Val::U8(42), "u8", 1, Primitive::I32),
        (Val::S16(-4242), "s16", 2, Primitive::I32),
        (Val::U16(4242), "u16", 2, Primitive::I32),
        (Val::S32(-314159265), "s32", 4, Primitive::I32),
        (Val::U32(314159265), "u32", 4, Primitive::I32),
        (Val::S64(-31415926535897), "s64", 8, Primitive::I64),
        (Val::U64(31415926535897), "u64", 8, Primitive::I64),
        (
            Val::Float32(3.14159265_f32.to_bits()),
            "float32",
            4,
            Primitive::F32,
        ),
        (
            Val::Float64(3.14159265_f64.to_bits()),
            "float64",
            8,
            Primitive::F64,
        ),
        (Val::Char('🦀'), "char", 4, Primitive::I32),
    ] {
        let component = Component::new(
            &engine,
            make_echo_component_with_primitive(ty, size, primitive),
        )?;
        let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

        let output = instance
            .get_func(&mut store, "echo")
            .unwrap()
            .call(&mut store, &[input.clone()])?;

        assert_eq!(input, output);
    }

    Ok(())
}
