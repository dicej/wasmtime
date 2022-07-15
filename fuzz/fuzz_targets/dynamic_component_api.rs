#![no_main]

use libfuzzer_sys::fuzz_target;
use wasmtime_fuzzing::oracles;

fuzz_target!(|bytes: &[u8]| {
    match oracles::dynamic_component_api_case(bytes) {
        Ok(()) | Err(arbitrary::Error::NotEnoughData) => (),
        Err(error) => panic!("{}", error),
    }
});
