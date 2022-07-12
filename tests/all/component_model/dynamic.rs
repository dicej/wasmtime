use super::make_echo_component;
use anyhow::Result;
use wasmtime::component::{Component, Linker, Val};
use wasmtime::Store;

#[test]
fn primitives() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("u32", 4))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;

    let input = Val::U32(314159265);
    let output = instance
        .get_func(&mut store, "echo")
        .unwrap()
        .call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}
