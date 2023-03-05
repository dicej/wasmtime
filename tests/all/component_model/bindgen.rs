use super::engine;
use anyhow::Result;
use wasmtime::{
    component::{Component, Linker},
    Store,
};

mod results;

mod no_imports {
    use super::*;

    wasmtime::component::bindgen!({
        inline: "
            default world no-imports {
                export foo: interface {
                    foo: func()
                }

                export bar: func()
            }
        ",
    });

    #[test]
    fn run() -> Result<()> {
        let engine = engine();

        let component = Component::new(
            &engine,
            r#"
                (component
                    (core module $m
                        (func (export ""))
                    )
                    (core instance $i (instantiate $m))

                    (func $f (export "bar") (canon lift (core func $i "")))

                    (instance $i (export "foo" (func $f)))
                    (export "foo" (instance $i))
                )
            "#,
        )?;

        let linker = Linker::new(&engine);
        let mut store = Store::new(&engine, ());
        let (no_imports, _) = NoImports::instantiate(&mut store, &component, &linker)?;
        no_imports.call_bar(&mut store)?;
        no_imports.foo().call_foo(&mut store)?;
        Ok(())
    }
}

mod one_import {
    use super::*;

    wasmtime::component::bindgen!({
        inline: "
            default world one-import {
                import foo: interface {
                    foo: func()
                }

                export bar: func()
            }
        ",
    });

    #[test]
    fn run() -> Result<()> {
        let engine = engine();

        let component = Component::new(
            &engine,
            r#"
                (component
                    (import "foo" (instance $i
                        (export "foo" (func))
                    ))
                    (core module $m
                        (import "" "" (func))
                        (export "" (func 0))
                    )
                    (core func $f (canon lower (func $i "foo")))
                    (core instance $i (instantiate $m
                        (with "" (instance (export "" (func $f))))
                    ))

                    (func $f (export "bar") (canon lift (core func $i "")))
                )
            "#,
        )?;

        #[derive(Default)]
        struct MyImports {
            hit: bool,
        }

        impl foo::Host for MyImports {
            fn foo(&mut self) -> Result<()> {
                self.hit = true;
                Ok(())
            }
        }

        let mut linker = Linker::new(&engine);
        foo::add_to_linker(&mut linker, |f: &mut MyImports| f)?;
        let mut store = Store::new(&engine, MyImports::default());
        let (one_import, _) = OneImport::instantiate(&mut store, &component, &linker)?;
        one_import.call_bar(&mut store)?;
        assert!(store.data().hit);
        Ok(())
    }
}

mod wildcards {
    use super::*;
    use component_test_util::TypedFuncExt;

    // We won't actually use any code this produces, but it's here to assert that the macro doesn't get confused by
    // wildcards:
    wasmtime::component::bindgen!({
        inline: "
            default world wildcards {
                import imports: interface {
                    *: func() -> u32
                }
                export exports: interface {
                    *: func() -> u32
                }
            }
        "
    });

    fn lambda(
        v: u32,
    ) -> Box<dyn Fn(wasmtime::StoreContextMut<'_, ()>, ()) -> Result<(u32,)> + Send + Sync> {
        Box::new(move |_, _| Ok((v,)))
    }

    #[test]
    fn run() -> Result<()> {
        let engine = engine();

        let component = Component::new(
            &engine,
            r#"
                (component
                    (import "imports" (instance $i
                        (export "a" (func (result u32)))
                        (export "b" (func (result u32)))
                        (export "c" (func (result u32)))
                    ))
                    (core module $m
                        (import "" "a" (func (result i32)))
                        (import "" "b" (func (result i32)))
                        (import "" "c" (func (result i32)))
                        (export "x" (func 0))
                        (export "y" (func 1))
                        (export "z" (func 2))
                    )
                    (core func $a (canon lower (func $i "a")))
                    (core func $b (canon lower (func $i "b")))
                    (core func $c (canon lower (func $i "c")))
                    (core instance $j (instantiate $m
                        (with "" (instance
                            (export "a" (func $a))
                            (export "b" (func $b))
                            (export "c" (func $c))
                        ))
                    ))
                    (func $x (result u32) (canon lift (core func $j "x")))
                    (func $y (result u32) (canon lift (core func $j "y")))
                    (func $z (export "z") (result u32) (canon lift (core func $j "z")))
                    (instance $k
                       (export "x" (func $x))
                       (export "y" (func $y))
                       (export "z" (func $z))
                    )
                    (export "exports" (instance $k))
                )
            "#,
        )?;

        let mut linker = Linker::<()>::new(&engine);
        let mut instance = linker.instance("imports")?;
        // In this simple test case, we don't really need to use `Component::names` to discover the imported
        // function names, but in the general case, we would, so we verify they're all present.
        for name in component.names("imports") {
            instance.func_wrap(
                name,
                match name {
                    "a" => lambda(42),
                    "b" => lambda(43),
                    "c" => lambda(44),
                    _ => unreachable!(),
                },
            )?;
        }

        let mut store = Store::new(&engine, ());
        let instance = linker.instantiate(&mut store, &component)?;
        let (mut x, mut y, mut z) = (None, None, None);

        {
            let mut exports = instance.exports(&mut store);
            let mut exports = exports.instance("exports").unwrap();
            // In this simple test case, we don't really need to use `ExportInstance::funcs` to discover the
            // exported function names, but in the general case, we would, so we verify they're all present.
            for (name, func) in exports.funcs() {
                match name {
                    "x" => x = Some(func),
                    "y" => y = Some(func),
                    "z" => z = Some(func),
                    _ => unreachable!(),
                }
            }
        };

        let (x, y, z) = (
            x.unwrap().typed::<(), (u32,)>(&store)?,
            y.unwrap().typed::<(), (u32,)>(&store)?,
            z.unwrap().typed::<(), (u32,)>(&store)?,
        );

        assert_eq!(42, x.call_and_post_return(&mut store, ())?.0);
        assert_eq!(43, y.call_and_post_return(&mut store, ())?.0);
        assert_eq!(44, z.call_and_post_return(&mut store, ())?.0);

        Ok(())
    }
}
