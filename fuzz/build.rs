fn main() -> anyhow::Result<()> {
    #[cfg(feature = "component-model")]
    component::generate_static_api_tests()?;

    Ok(())
}

#[cfg(feature = "component-model")]
mod component {
    use anyhow::{anyhow, Context, Result};
    use arbitrary::{Arbitrary, Unstructured};
    use proc_macro2::TokenStream;
    use quote::{format_ident, quote};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::env;
    use std::fmt::Write;
    use std::fs;
    use std::ops::DerefMut;
    use std::path::PathBuf;
    use std::process::Command;
    use wasmtime_fuzzing::generators::component_types::{self, Declarations, TestCase};

    pub fn generate_static_api_tests() -> Result<()> {
        println!("cargo:rerun-if-changed=build.rs");
        let out_dir = PathBuf::from(
            env::var_os("OUT_DIR").expect("The OUT_DIR environment variable must be set"),
        );

        let mut out = String::new();
        write_static_api_tests(&mut out)?;

        let output = out_dir.join("static_component_api.rs");
        fs::write(&output, out)?;

        drop(Command::new("rustfmt").arg(&output).status());

        Ok(())
    }

    fn write_static_api_tests(out: &mut String) -> Result<()> {
        let seed = if let Ok(seed) = env::var("WASMTIME_FUZZ_SEED") {
            seed.parse::<u64>()
                .with_context(|| anyhow!("expected u64 in WASMTIME_FUZZ_SEED"))?
        } else {
            StdRng::from_entropy().gen()
        };

        let mut rng = StdRng::seed_from_u64(seed);

        const TEST_CASE_COUNT: usize = 100;

        let mut tests = TokenStream::new();

        let name_counter = &mut 0;

        let mut declarations = TokenStream::new();

        for index in 0..TEST_CASE_COUNT {
            let mut bytes = vec![0u8; rng.gen_range(1000..2000)];
            rng.fill(bytes.deref_mut());

            let case = TestCase::arbitrary(&mut Unstructured::new(&bytes))?;

            let Declarations {
                types,
                params,
                result,
                import_and_export,
            } = case.declarations();

            let test = format_ident!("static_api_test{}", case.params.len());

            let rust_params = case
                .params
                .iter()
                .map(|ty| {
                    let ty = component_types::rust_type(&ty, name_counter, &mut declarations);
                    quote!(#ty,)
                })
                .collect::<TokenStream>();

            let rust_result =
                component_types::rust_type(&case.result, name_counter, &mut declarations);

            let test = quote!(#index => component_types::#test::<#rust_params #rust_result>(
                &mut input,
                &Declarations {
                    types: #types,
                    params: #params,
                    result: #result,
                    import_and_export: #import_and_export
                }
            ),);

            tests.extend(test);
        }

        let module = quote! {
            #[allow(unused_imports)]
            fn target(bytes: &[u8]) -> arbitrary::Result<()> {
                use anyhow::Result;
                use arbitrary::{Unstructured, Arbitrary};
                use component_test_util::{self, Float32, Float64};
                use std::sync::{Arc, Once};
                use wasmtime::component::{ComponentType, Lift, Lower};
                use wasmtime_fuzzing::generators::component_types::{self, Declarations};

                const SEED: u64 = #seed;

                static ONCE: Once = Once::new();

                ONCE.call_once(|| {
                    eprintln!(
                        "Seed {SEED} was used to generate static component API fuzz tests.\n\
                         Set WASMTIME_FUZZ_SEED env variable at build time to reproduce."
                    );
                });

                #declarations

                let mut input = Unstructured::new(bytes);
                match input.int_in_range(0..=(#TEST_CASE_COUNT-1))? {
                    #tests
                    _ => unreachable!()
                }
            }

            libfuzzer_sys::fuzz_target!(|bytes: &[u8]| {
                match target(bytes) {
                    Ok(()) | Err(arbitrary::Error::NotEnoughData) => (),
                    Err(error) => panic!("{}", error),
                }
            });
        };

        write!(out, "{module}")?;

        Ok(())
    }
}
