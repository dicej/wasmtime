use crate::{
    component::{func::FuncData, Resource},
    store::Stored,
    AsContextMut, StoreContextMut, ValRaw,
};
use anyhow::{anyhow, Result};
use futures::{
    future::FutureExt,
    stream::{FuturesUnordered, ReadyChunks, StreamExt},
};
use once_cell::sync::Lazy;
use std::any::Any;
use std::future::Future;
use std::mem::{self, MaybeUninit};
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use sync_resource_table::ResourceTable;
use wasmtime_environ::component::TypeFuncIndex;
use wasmtime_runtime::{component::VMComponentContext, VMOpaqueContext};

mod sync_resource_table;

type BoxFuture<T> = Pin<
    Box<
        dyn Future<Output = (u32, Box<dyn FnOnce(StoreContextMut<'_, T>) -> Result<()>>)>
            + Send
            + Sync
            + 'static,
    >,
>;

pub(crate) struct Task;

fn dummy_waker() -> Waker {
    struct DummyWaker;

    impl Wake for DummyWaker {
        fn wake(self: Arc<Self>) {}
    }

    static WAKER: Lazy<Arc<DummyWaker>> = Lazy::new(|| Arc::new(DummyWaker));

    WAKER.clone().into()
}

pub fn for_any<F, R, T>(fun: F) -> F
where
    F: FnOnce(StoreContextMut<T>) -> R + 'static,
    R: 'static,
{
    fun
}

pub(crate) fn first_poll<T, R: Send + 'static>(
    mut store: StoreContextMut<T>,
    future: impl Future<Output = impl FnOnce(StoreContextMut<T>) -> Result<R> + 'static>
        + Send
        + Sync
        + 'static,
    lower: impl FnOnce(StoreContextMut<T>, R) -> Result<()> + Send + Sync + 'static,
) -> Result<Option<Resource<Task>>> {
    // TODO: make this a child of the current export call
    let task = store.concurrent_state().table.push(Task)?;
    let rep = task.rep();
    let mut future = Box::pin(future.map(move |fun| {
        (
            rep,
            Box::new(for_any(move |mut store| {
                let result = fun(store.as_context_mut())?;
                lower(store, result)
            })) as Box<dyn FnOnce(StoreContextMut<T>) -> Result<()>>,
        )
    })) as BoxFuture<T>;

    Ok(
        match future
            .as_mut()
            .poll(&mut Context::from_waker(&dummy_waker()))
        {
            Poll::Ready((_, fun)) => {
                store.concurrent_state().table.delete(task)?;
                fun(store)?;
                None
            }
            Poll::Pending => {
                store.concurrent_state().futures.get_mut().push(future);
                Some(task)
            }
        },
    )
}

pub(crate) async fn poll_loop<T>(store: &mut StoreContextMut<'_, T>) -> Result<()> {
    const EVENT_CALL_DONE: i32 = 2;

    while store.concurrent_state().result.is_none() {
        if let Some(ready) = store.concurrent_state().futures.next().await {
            for (rep, fun) in ready {
                fun(store.as_context_mut())?;
                let (func, context) = store.concurrent_state().context.unwrap();
                let callback = store.0[func].options.callback.unwrap();
                let params = &mut [
                    ValRaw::i32(context),
                    ValRaw::i32(EVENT_CALL_DONE),
                    ValRaw::u32(rep),
                    ValRaw::i32(0),
                ];
                unsafe {
                    crate::Func::call_unchecked_raw(
                        store,
                        callback,
                        params.as_mut_ptr(),
                        params.len(),
                    )?;
                }
                // TODO: look at return value and decide what to do if it's zero (i.e. not yet done)
            }
        } else {
            break;
        }
    }

    Ok(())
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
    pub(crate) context: Option<(Stored<FuncData>, i32)>,
    futures: ReadyChunks<FuturesUnordered<BoxFuture<T>>>,
    table: ResourceTable,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            lower_params: None,
            lift_result: None,
            result: None,
            context: None,
            table: ResourceTable::new(),
            futures: FuturesUnordered::new().ready_chunks(1024),
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
            let storage = std::slice::from_raw_parts(storage, storage_len);
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
