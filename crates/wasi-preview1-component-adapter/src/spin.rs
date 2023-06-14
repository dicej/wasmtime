#![allow(clashing_extern_declarations)]

#[link(wasm_import_module = "__main_module__")]
extern "C" {
    pub fn canonical_abi_realloc(
        old_ptr: *mut u8,
        old_len: usize,
        align: usize,
        new_len: usize,
    ) -> *mut u8;
}

unsafe fn dealloc(ptr: i32, size: usize, align: usize) {
    #[link(wasm_import_module = "__main_module__")]
    extern "C" {
        fn canonical_abi_free(ptr: *mut u8, len: usize, align: usize);
    }

    canonical_abi_free(ptr as _, size, align);
}

macro_rules! wrap_export {
    ($export_name:literal $name:ident $import_name:literal $( $arg:ident )*) => {
        #[export_name = $export_name]
        unsafe extern "C" fn $name($( $arg: i32 ),*) -> i32 {
            #[link(wasm_import_module = "__main_module__")]
            extern "C" {
                #[cfg_attr(target_arch = "wasm32", link_name = $import_name)]
                fn wit_import($( $arg: i32 ),*) -> i32;
            }
            wit_import($( $arg ),*)
        }
    }
}

wrap_export!("fermyon:spin/inbound-redis#handle-message" inbound_redis_handle_message "handle-redis-message"
             a0 a1);

wrap_export!("fermyon:spin/inbound-http#handle-request" inbound_http_handle_request "handle-http-request"
             a0 a1 a2 a3 a4 a5 a6 a7 a8 a9);

#[doc(hidden)]
#[export_name = "cabi_post_fermyon:spin/inbound-http#handle-request"]
#[allow(non_snake_case)]
unsafe extern "C" fn post_return_inbound_http_handle_request(arg0: i32) {
    match i32::from(*((arg0 + 4) as *const u8)) {
        0 => (),
        _ => {
            let base0 = *((arg0 + 8) as *const i32);
            let len0 = *((arg0 + 12) as *const i32);
            for i in 0..len0 {
                let base = base0 + i * 16;
                {
                    dealloc(
                        *((base + 0) as *const i32),
                        (*((base + 4) as *const i32)) as usize,
                        1,
                    );
                    dealloc(
                        *((base + 8) as *const i32),
                        (*((base + 12) as *const i32)) as usize,
                        1,
                    );
                }
            }
            dealloc(base0, (len0 as usize) * 16, 4);
        }
    }
    match i32::from(*((arg0 + 16) as *const u8)) {
        0 => (),
        _ => {
            let base1 = *((arg0 + 20) as *const i32);
            let len1 = *((arg0 + 24) as *const i32);
            dealloc(base1, (len1 as usize) * 1, 1);
        }
    }
}

/// Wrap the adapter imports in new names
///
/// `export_name` - is what the "main" module (i.e., the module that we are adapting) have as imports
/// The `import_module` and `import_name` are what the adapted module will then import
macro_rules! wrap_import {
    ($export_name:literal $name:ident $import_module:literal $import_name:literal $( $arg:ident : $arg_type:path )*) => {
        #[export_name = $export_name]
        unsafe extern "C" fn $name($( $arg: $arg_type ),*) {
            #[link(wasm_import_module = $import_module)]
            extern "C" {
                #[cfg_attr(target_arch = "wasm32", link_name = $import_name)]
                fn wit_import($( $arg: $arg_type ),*);
            }
            super::State::with(|state| {
                state.import_alloc.with_main(|| wit_import($( $arg ),*));
                Ok(())
            });
        }
    }
}

wrap_import!("wasi-outbound-http:request" wasi_outbound_http_request "fermyon:spin/http" "send-request"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32 a7:i32 a8:i32 a9:i32 a10:i32);

wrap_import!("spin-config:get-config" config_get_config "fermyon:spin/config" "get-config"
             a0:i32 a1:i32 a2:i32);

wrap_import!("outbound-redis:publish" outbound_redis_publish "fermyon:spin/redis" "publish"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-redis:set" outbound_redis_set "fermyon:spin/redis" "set"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-redis:get" outbound_redis_get "fermyon:spin/redis" "get"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32);

wrap_import!("outbound-redis:incr" outbound_redis_incr "fermyon:spin/redis" "incr"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32);

wrap_import!("outbound-redis:del" outbound_redis_del "fermyon:spin/redis" "del"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32);

wrap_import!("outbound-redis:sadd" outbound_redis_sadd "fermyon:spin/redis" "sadd"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-redis:smembers" outbound_redis_smembers "fermyon:spin/redis" "smembers"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32);

wrap_import!("outbound-redis:srem" outbound_redis_srem "fermyon:spin/redis" "srem"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-redis:execute" outbound_redis_execute "fermyon:spin/redis" "execute"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-pg:query" outbound_pg_query "fermyon:spin/postgres" "query"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-pg:execute" outbound_pg_execute "fermyon:spin/postgres" "execute"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-mysql:query" outbound_mysql_query "fermyon:spin/mysql" "query"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("outbound-mysql:execute" outbound_mysql_execute "fermyon:spin/mysql" "execute"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32 a6:i32);

wrap_import!("sqlite:open" sqlite_open "fermyon:spin/sqlite" "open" a0:i32 a1:i32 a2:i32);

wrap_import!("sqlite:execute" sqlite_execute "fermyon:spin/sqlite" "execute" a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32);

wrap_import!("sqlite:close" sqlite_close "fermyon:spin/sqlite" "close" a0:i32);

wrap_import!("key-value:open" key_value_open "fermyon:spin/key-value" "open"
             a0:i32 a1:i32 a2:i32);

wrap_import!("key-value:get" key_value_get "fermyon:spin/key-value" "get"
             a0:i32 a1:i32 a2:i32 a3:i32);

wrap_import!("key-value:set" key_value_set "fermyon:spin/key-value" "set"
             a0:i32 a1:i32 a2:i32 a3:i32 a4:i32 a5:i32);

wrap_import!("key-value:delete" key_value_delete "fermyon:spin/key-value" "delete"
             a0:i32 a1:i32 a2:i32 a3:i32);

wrap_import!("key-value:exists" key_value_exists "fermyon:spin/key-value" "exists"
             a0:i32 a1:i32 a2:i32 a3:i32);

wrap_import!("key-value:get-keys" key_value_get_keys "fermyon:spin/key-value" "get-keys"
             a0:i32 a1:i32);

wrap_import!("key-value:close" key_value_close "fermyon:spin/key-value" "close"
             a0:i32);

wrap_import!("llm:infer" llm_infer "fermyon:spin/llm" "infer" 
            a0: i32 a1: i32 a2: i32 a3: i32 a4: i32 a5: i32 a6: f32 a7: i32 a8: f32 a9: i32 a10: f32 a11: i32);

wrap_import!("llm:generate-embeddings" llm_generate_embeddings "fermyon:spin/llm" "generate-embeddings" 
            a0:i32 a1:i32 a2:i32 a3:i32 a4:i32);
