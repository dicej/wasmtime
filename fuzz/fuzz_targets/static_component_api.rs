#![no_main]

use libfuzzer_sys::fuzz_target;

include!(concat!(env!("OUT_DIR"), "/static_component_api.rs"));

libfuzzer_sys::fuzz_target!(|bytes: &[u8]| {
    let mut input = Unstructured::new(bytes);

    match target(bytes) {
        Ok(()) | Err(arbitrary::Error::NotEnoughData) => (),
        Err(error) => panic!("{}", error),
    }
});
