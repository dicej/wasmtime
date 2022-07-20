use super::{make_echo_component, make_echo_component_with_params, Type};
use anyhow::Result;
use wasmtime::component::{self, Component, Linker, Val};
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
        let func = instance.get_func(&mut store, "echo").unwrap();
        let output = func.call(&mut store, &[input.clone()])?;

        assert_eq!(input, output);
    }

    // Sad path: type mismatch

    let component = Component::new(
        &engine,
        make_echo_component_with_params("float64", &[Type::F64]),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();

    assert!(func.call(&mut store, &[Val::U64(42)]).is_err());

    // Sad path: arity mismatch (too many)

    assert!(func
        .call(
            &mut store,
            &[
                Val::Float64(3.14159265_f64.to_bits()),
                Val::Float64(3.14159265_f64.to_bits())
            ]
        )
        .is_err());

    // Sad path: arity mismatch (too few)

    assert!(func.call(&mut store, &[]).is_err());

    Ok(())
}

#[test]
fn strings() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("string", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let input = Val::String(Box::from("hello, component!"));
    let output = func.call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}

#[test]
fn lists() -> Result<()> {
    let engine = super::engine();
    let mut store = Store::new(&engine, ());

    let component = Component::new(&engine, make_echo_component("(list u32)", 8))?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let input = ty.new_list(Box::new([
        Val::U32(32343),
        Val::U32(79023439),
        Val::U32(2084037802),
    ]))?;
    let output = func.call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    // Sad path: type mismatch

    assert!(ty
        .new_list(Box::new([
            Val::U32(32343),
            Val::U32(79023439),
            Val::Float32(3.14159265_f32.to_bits()),
        ]))
        .is_err());

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
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let inner_type = if let component::Type::Record(handle) = ty {
        handle.fields().nth(2).unwrap().ty
    } else {
        unreachable!()
    };
    let input = ty.new_record(
        [
            ("A", Val::U32(32343)),
            ("B", Val::Float64(3.14159265_f64.to_bits())),
            (
                "C",
                inner_type.new_record(
                    [("D", Val::Bool(false)), ("E", Val::U32(2084037802))].into_iter(),
                )?,
            ),
        ]
        .into_iter(),
    )?;
    let output = func.call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    // Sad path: type mismatch

    assert!(ty
        .new_record(
            [
                ("A", Val::S32(32343)),
                ("B", Val::Float64(3.14159265_f64.to_bits())),
                (
                    "C",
                    inner_type.new_record(
                        [("D", Val::Bool(false)), ("E", Val::U32(2084037802))].into_iter(),
                    )?,
                ),
            ]
            .into_iter(),
        )
        .is_err());

    // Sad path: too many fields

    assert!(ty
        .new_record(
            [
                ("A", Val::U32(32343)),
                ("B", Val::Float64(3.14159265_f64.to_bits())),
                (
                    "C",
                    inner_type.new_record(
                        [("D", Val::Bool(false)), ("E", Val::U32(2084037802))].into_iter(),
                    )?,
                ),
                ("F", Val::Unit)
            ]
            .into_iter(),
        )
        .is_err());

    // Sad path: too few fields

    assert!(ty
        .new_record(
            [
                ("A", Val::U32(32343)),
                ("B", Val::Float64(3.14159265_f64.to_bits()))
            ]
            .into_iter(),
        )
        .is_err());

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
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let input = ty.new_variant("B", Val::Float64(3.14159265_f64.to_bits()))?;
    let output = func.call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    // Sad path: type mismatch

    assert!(ty.new_variant("B", Val::U64(314159265)).is_err());

    // Sad path: unknown discriminant

    assert!(ty.new_variant("D", Val::U64(314159265)).is_err());

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
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let input = ty.new_flags(&["B", "D"])?;
    let output = func.call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    // Sad path: unknown flags

    assert!(ty.new_flags(&["B", "D", "F"]).is_err());

    Ok(())
}

#[test]
fn everything() -> Result<()> {
    // This serves to test both nested types and storing parameters on the heap (i.e. exceeding `MAX_STACK_PARAMS`)

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
                (field "AA" (option u32))
                (field "BB" (expected string string))
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
                Type::U8,
                Type::I32,
                Type::U8,
                Type::I32,
                Type::I32,
            ],
        ),
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let func = instance.get_func(&mut store, "echo").unwrap();
    let ty = &func.params(&store)[0];
    let types = if let component::Type::Record(handle) = ty {
        handle.fields().map(|field| field.ty).collect::<Box<[_]>>()
    } else {
        unreachable!()
    };
    let (b_type, c_type, f_type, j_type, y_type, z_type, aa_type, bb_type) = (
        &types[1], &types[2], &types[3], &types[4], &types[14], &types[15], &types[16], &types[17],
    );
    let f_element_type = &if let component::Type::List(handle) = &f_type {
        handle.ty()
    } else {
        unreachable!()
    };
    let input = ty.new_record(
        [
            ("A", Val::U32(32343)),
            ("B", b_type.new_enum("2")?),
            (
                "C",
                c_type.new_record(
                    [("D", Val::Bool(false)), ("E", Val::U32(2084037802))].into_iter(),
                )?,
            ),
            (
                "F",
                f_type.new_list(Box::new([f_element_type.new_flags(&["G", "I"])?]))?,
            ),
            (
                "J",
                j_type.new_variant("L", Val::Float64(3.14159265_f64.to_bits()))?,
            ),
            ("P", Val::S8(42)),
            ("Q", Val::S16(4242)),
            ("R", Val::S32(42424242)),
            ("S", Val::S64(424242424242424242)),
            ("T", Val::Float32(3.14159265_f32.to_bits())),
            ("U", Val::Float64(3.14159265_f64.to_bits())),
            ("V", Val::String(Box::from("wow, nice types"))),
            ("W", Val::Char('🦀')),
            ("X", Val::Unit),
            (
                "Y",
                y_type.new_tuple(Box::new([Val::U32(42), Val::U32(24)]))?,
            ),
            (
                "Z",
                z_type.new_union(1, Val::Float64(3.14159265_f64.to_bits()))?,
            ),
            ("AA", aa_type.new_option(Some(Val::U32(314159265)))?),
            (
                "BB",
                bb_type.new_expected(Ok(Val::String(Box::from("no problem"))))?,
            ),
        ]
        .into_iter(),
    )?;
    let output = func.call(&mut store, &[input.clone()])?;

    assert_eq!(input, output);

    Ok(())
}
