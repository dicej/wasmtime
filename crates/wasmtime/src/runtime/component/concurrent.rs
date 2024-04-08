use crate::component::Resource;
use crate::{StoreContextMut, ValRaw};
use anyhow::{anyhow, Result};
use std::any::Any;
use std::future::Future;
use std::mem::{self, MaybeUninit};
use std::panic::{self, AssertUnwindSafe};
use wasmtime_environ::component::TypeFuncIndex;
use wasmtime_runtime::{component::VMComponentContext, VMOpaqueContext};

pub(crate) struct Task;

pub(crate) fn first_poll<T, R: Send + 'static>(
    store: StoreContextMut<T>,
    future: impl Future<Output = impl FnOnce(StoreContextMut<T>) -> Result<R> + 'static>
        + Send
        + 'static,
) -> Result<Result<R, Resource<Task>>> {
    _ = (store, future);
    todo!()
}

pub(crate) async fn poll_loop<T>(store: &mut StoreContextMut<'_, T>) -> Result<()> {
    _ = store;
    todo!()
}

pub struct ConcurrentState<T> {
    pub(crate) lower_params: Option<
        Box<
            dyn FnOnce(&mut StoreContextMut<T>, &mut [MaybeUninit<ValRaw>]) -> Result<()>
                + Send
                + Sync,
        >,
    >,
    pub(crate) lift_result: Option<
        Box<
            dyn FnOnce(&mut StoreContextMut<T>, &[ValRaw]) -> Result<Box<dyn Any + Send + Sync>>
                + Send
                + Sync,
        >,
    >,
    pub(crate) result: Option<Box<dyn Any + Send + Sync>>,
    pub(crate) context: i32,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            lower_params: None,
            lift_result: None,
            result: None,
            context: 0,
        }
    }
}

pub extern "C" fn async_start<T>(
    cx: *mut VMOpaqueContext,
    _ty: TypeFuncIndex,
    storage: *mut MaybeUninit<ValRaw>,
    storage_len: usize,
) {
    unsafe {
        handle_result(|| {
            let storage = std::slice::from_raw_parts_mut(storage, storage_len);
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let lower = cx
                .concurrent_state()
                .lower_params
                .take()
                .ok_or_else(|| anyhow!("call.start called more than once"))?;
            lower(&mut cx, storage)
        })
    }
}

pub extern "C" fn async_return<T>(
    cx: *mut VMOpaqueContext,
    _ty: TypeFuncIndex,
    storage: *mut MaybeUninit<ValRaw>,
    storage_len: usize,
) {
    unsafe {
        handle_result(|| {
            let storage = std::slice::from_raw_parts_mut(storage, storage_len);
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let lift = cx
                .concurrent_state()
                .lift_result
                .take()
                .ok_or_else(|| anyhow!("call.return called more than once"))?;

            assert!(cx.concurrent_state().result.is_none());

            let result = lift(
                &mut cx,
                mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(storage),
            )?;
            cx.concurrent_state().result = Some(result);

            Ok(())
        })
    }
}

unsafe fn handle_result(func: impl FnOnce() -> Result<()>) {
    match panic::catch_unwind(AssertUnwindSafe(func)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => crate::trap::raise(e),
        Err(e) => wasmtime_runtime::resume_panic(e),
    }
}
