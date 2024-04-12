use crate::{AsContextMut, StoreContextMut, ValRaw};
use anyhow::{anyhow, Result};
use futures::{
    future::FutureExt,
    stream::{FuturesUnordered, ReadyChunks, StreamExt},
};
use once_cell::sync::Lazy;
use std::{
    any::Any,
    future::Future,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    panic::{self, AssertUnwindSafe},
    pin::{pin, Pin},
    ptr::NonNull,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
use task_table::TaskTable;
use wasmtime_environ::component::TypeFuncIndex;
use wasmtime_runtime::{component::VMComponentContext, SendSyncPtr, VMFuncRef, VMOpaqueContext};

mod task_table;

const STATUS_NOT_STARTED: u32 = 0;
const STATUS_PARAMS_READ: u32 = 1;
const STATUS_RESULTS_WRITTEN: u32 = 2;
const STATUS_DONE: u32 = 3;

const EVENT_CALL_STARTED: u32 = 0;
const EVENT_CALL_RETURNED: u32 = 1;
const EVENT_CALL_DONE: u32 = 2;

pub(crate) struct TaskId<T> {
    rep: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> TaskId<T> {
    fn new(rep: u32) -> Self {
        Self {
            rep,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for TaskId<T> {
    fn clone(&self) -> Self {
        Self::new(self.rep)
    }
}

impl<T> Copy for TaskId<T> {}

impl<T> TaskId<T> {
    pub fn rep(&self) -> u32 {
        self.rep
    }
}

type HostTaskFuture<T> = Pin<
    Box<
        dyn Future<Output = (u32, Box<dyn FnOnce(StoreContextMut<'_, T>) -> Result<()>>)>
            + Send
            + Sync
            + 'static,
    >,
>;

struct HostTask<T> {
    caller: TaskId<T>,
}

struct Caller<T> {
    task: TaskId<T>,
    store_call: SendSyncPtr<VMFuncRef>,
    call: u32,
}

struct GuestTask<T> {
    lower_params: Option<Lower<T>>,
    lift_result: Option<Lift<T>>,
    result: Option<LiftedResult>,
    callback: Option<(SendSyncPtr<VMFuncRef>, u32)>,
    caller: Option<Caller<T>>,
}

enum Task<T> {
    Host(HostTask<T>),
    Guest(GuestTask<T>),
}

impl<T> Task<T> {
    fn unwrap_guest(self) -> GuestTask<T> {
        if let Self::Guest(task) = self {
            task
        } else {
            unreachable!()
        }
    }

    fn unwrap_guest_ref(&self) -> &GuestTask<T> {
        if let Self::Guest(task) = self {
            task
        } else {
            unreachable!()
        }
    }

    fn unwrap_guest_mut(&mut self) -> &mut GuestTask<T> {
        if let Self::Guest(task) = self {
            task
        } else {
            unreachable!()
        }
    }

    fn unwrap_host(self) -> HostTask<T> {
        if let Self::Host(task) = self {
            task
        } else {
            unreachable!()
        }
    }
}

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
) -> Result<Option<TaskId<T>>> {
    let caller = store.concurrent_state().guest_task.unwrap();
    let task = store
        .concurrent_state()
        .table
        .push_child(Task::Host(HostTask { caller }), caller)?;
    let mut future = Box::pin(future.map(move |fun| {
        (
            task.rep,
            Box::new(for_any(move |mut store| {
                let result = fun(store.as_context_mut())?;
                lower(store, result)
            })) as Box<dyn FnOnce(StoreContextMut<T>) -> Result<()>>,
        )
    })) as HostTaskFuture<T>;

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

fn maybe_send_event<T>(
    store: &mut StoreContextMut<'_, T>,
    guest_task: TaskId<T>,
    event: u32,
    call: TaskId<T>,
) -> Result<()> {
    if let Some((callback, context)) = store
        .concurrent_state()
        .table
        .get(guest_task)?
        .unwrap_guest_ref()
        .callback
    {
        let old_task = store.concurrent_state().guest_task.replace(guest_task);
        let params = &mut [
            ValRaw::u32(context),
            ValRaw::u32(event),
            ValRaw::u32(call.rep),
            ValRaw::i32(0),
        ];
        unsafe {
            crate::Func::call_unchecked_raw(
                store,
                callback.as_non_null(),
                params.as_mut_ptr(),
                params.len(),
            )?;
        }
        let done = params[0].get_u32() != 0;
        if done {
            if let Some(next) = store
                .concurrent_state()
                .table
                .get(guest_task)?
                .unwrap_guest_ref()
                .caller
                .as_ref()
                .map(|c| c.task)
            {
                store.concurrent_state().table.delete(guest_task)?;
                maybe_send_event(store, next, EVENT_CALL_DONE, guest_task)?;
            }
        }
        store.concurrent_state().guest_task = old_task;
    }
    Ok(())
}

async fn poll_loop<T>(store: &mut StoreContextMut<'_, T>) -> Result<()> {
    let task = store.concurrent_state().guest_task.unwrap();
    while store
        .concurrent_state()
        .table
        .get(task)?
        .unwrap_guest_ref()
        .result
        .is_none()
    {
        if let Some(ready) = store.concurrent_state().futures.next().await {
            for (task, fun) in ready {
                let task = TaskId::new(task);
                fun(store.as_context_mut())?;
                let caller = store
                    .concurrent_state()
                    .table
                    .delete(task)?
                    .unwrap_host()
                    .caller;
                maybe_send_event(store, caller, EVENT_CALL_DONE, task)?;
            }
        } else {
            break;
        }
    }

    Ok(())
}

type Lower<T> = Box<
    dyn FnOnce(&mut StoreContextMut<T>, &mut [MaybeUninit<ValRaw>]) -> Result<()> + Send + Sync,
>;

type Lift<T> = Box<
    dyn FnOnce(&mut StoreContextMut<T>, &[ValRaw]) -> Result<Option<Box<dyn Any + Send + Sync>>>
        + Send
        + Sync,
>;

type LiftedResult = Box<dyn Any + Send + Sync>;

pub struct ConcurrentState<T> {
    guest_task: Option<TaskId<T>>,
    futures: ReadyChunks<FuturesUnordered<HostTaskFuture<T>>>,
    table: TaskTable<T>,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            guest_task: None,
            table: TaskTable::new(),
            futures: FuturesUnordered::new().ready_chunks(1024),
        }
    }
}

pub(crate) extern "C" fn async_start<T>(
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
            let guest_task = cx.concurrent_state().guest_task.unwrap();
            let lower = cx
                .concurrent_state()
                .table
                .get_mut(guest_task)?
                .unwrap_guest_mut()
                .lower_params
                .take()
                .ok_or_else(|| anyhow!("call.start called more than once"))?;
            lower(&mut cx, storage)
        })
    }
}

pub(crate) extern "C" fn async_return<T>(
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
            let guest_task = cx.concurrent_state().guest_task.unwrap();
            let lift = cx
                .concurrent_state()
                .table
                .get_mut(guest_task)?
                .unwrap_guest_mut()
                .lift_result
                .take()
                .ok_or_else(|| anyhow!("call.return called more than once"))?;

            assert!(cx
                .concurrent_state()
                .table
                .get(guest_task)?
                .unwrap_guest_ref()
                .result
                .is_none());

            let result = lift(
                &mut cx,
                mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(storage),
            )?;
            cx.concurrent_state()
                .table
                .get_mut(guest_task)?
                .unwrap_guest_mut()
                .result = result;

            Ok(())
        })
    }
}

unsafe fn handle_result<T>(func: impl FnOnce() -> Result<T>) -> T {
    match panic::catch_unwind(AssertUnwindSafe(func)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => crate::trap::raise(e),
        Err(e) => wasmtime_runtime::resume_panic(e),
    }
}

pub(crate) extern "C" fn async_enter<T>(
    cx: *mut VMOpaqueContext,
    start: *mut VMFuncRef,
    return_: *mut VMFuncRef,
    store_call: *mut VMFuncRef,
    params: u32,
    results: u32,
    call: u32,
    expect_retptr: u32,
) {
    unsafe {
        handle_result(|| {
            let expect_retptr = expect_retptr != 0;
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let start = SendSyncPtr::new(NonNull::new(start).unwrap());
            let return_ = SendSyncPtr::new(NonNull::new(return_).unwrap());
            let old_task = cx.concurrent_state().guest_task.take().unwrap();
            let old_task_rep = old_task.rep;
            let guest_task = cx.concurrent_state().table.push_child(
                Task::Guest(GuestTask {
                    lower_params: Some(Box::new(move |cx, dst| {
                        let mut src = [ValRaw::u32(params), ValRaw::u32(0)];
                        if expect_retptr {
                            src[1] = dst[0].assume_init();
                        }
                        crate::Func::call_unchecked_raw(
                            cx,
                            start.as_non_null(),
                            src.as_mut_ptr(),
                            if expect_retptr { 2 } else { 1 },
                        )?;
                        if !expect_retptr {
                            dst[0] = MaybeUninit::new(src[0]);
                        }
                        let task = cx.concurrent_state().guest_task.unwrap();
                        maybe_send_event(cx, TaskId::new(old_task_rep), EVENT_CALL_STARTED, task)?;
                        Ok(())
                    })),
                    lift_result: Some(Box::new(move |cx, src| {
                        let mut my_src = src.to_owned();
                        my_src.push(ValRaw::u32(results));
                        crate::Func::call_unchecked_raw(
                            cx,
                            return_.as_non_null(),
                            my_src.as_mut_ptr(),
                            my_src.len(),
                        )?;
                        let task = cx.concurrent_state().guest_task.unwrap();
                        maybe_send_event(cx, TaskId::new(old_task_rep), EVENT_CALL_RETURNED, task)?;
                        Ok(None)
                    })),
                    result: None,
                    callback: None,
                    caller: Some(Caller {
                        task: old_task,
                        store_call: SendSyncPtr::new(NonNull::new(store_call).unwrap()),
                        call,
                    }),
                }),
                old_task,
            )?;
            cx.concurrent_state().guest_task = Some(guest_task);

            Ok(())
        })
    }
}

pub(crate) extern "C" fn async_exit<T>(
    cx: *mut VMOpaqueContext,
    callback: *mut VMFuncRef,
    guest_context: u32,
) -> u32 {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());

            let guest_task = cx.concurrent_state().guest_task.take().unwrap();
            let (store_call, call, next) = {
                let caller = cx
                    .concurrent_state()
                    .table
                    .get(guest_task)?
                    .unwrap_guest_ref()
                    .caller
                    .as_ref()
                    .unwrap();
                (caller.store_call.as_non_null(), caller.call, caller.task)
            };
            cx.concurrent_state().guest_task = Some(next);

            let task = cx
                .concurrent_state()
                .table
                .get_mut(guest_task)?
                .unwrap_guest_mut();

            let status = if task.lower_params.is_some() {
                STATUS_NOT_STARTED
            } else if task.lift_result.is_some() {
                STATUS_PARAMS_READ
            } else if guest_context != 0 {
                STATUS_RESULTS_WRITTEN
            } else {
                STATUS_DONE
            };

            if guest_context != 0 {
                task.callback = Some((
                    SendSyncPtr::new(NonNull::new(callback).unwrap()),
                    guest_context,
                ));

                let mut src = [ValRaw::u32(call), ValRaw::u32(guest_task.rep)];
                crate::Func::call_unchecked_raw(&mut cx, store_call, src.as_mut_ptr(), src.len())?;
            } else {
                cx.concurrent_state().table.delete(guest_task)?;
            }

            Ok(status)
        })
    }
}

pub(crate) fn enter<T: Send>(
    mut store: StoreContextMut<T>,
    lower_params: Lower<T>,
    lift_result: Lift<T>,
) -> Result<()> {
    assert!(store.concurrent_state().guest_task.is_none());

    let guest_task = store.concurrent_state().table.push(Task::Guest(GuestTask {
        lower_params: Some(lower_params),
        lift_result: Some(lift_result),
        result: None,
        callback: None,
        caller: None,
    }))?;
    store.concurrent_state().guest_task = Some(guest_task);

    Ok(())
}

pub(crate) fn exit<T: Send, R: 'static>(
    mut store: StoreContextMut<T>,
    callback: NonNull<VMFuncRef>,
    guest_context: u32,
) -> Result<R> {
    if guest_context != 0 {
        let guest_task = store.concurrent_state().guest_task.unwrap();
        store
            .concurrent_state()
            .table
            .get_mut(guest_task)?
            .unwrap_guest_mut()
            .callback = Some((SendSyncPtr::new(callback), guest_context));

        {
            let async_cx = store.0.async_cx().expect("async cx");
            let mut future = pin!(poll_loop(&mut store));
            unsafe {
                async_cx.block_on(future.as_mut())??;
            }
        }
    }

    let guest_task = store.concurrent_state().guest_task.take().unwrap();
    if let Some(result) = store
        .concurrent_state()
        .table
        .delete(guest_task)?
        .unwrap_guest()
        .result
        .take()
    {
        Ok(*result.downcast().unwrap())
    } else {
        Err(anyhow!(crate::Trap::NoAsyncResult))
    }
}
