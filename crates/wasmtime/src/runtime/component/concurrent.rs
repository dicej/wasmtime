use crate::{AsContextMut, StoreContextMut, ValRaw};
use anyhow::{anyhow, Error, Result};
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
use wasmtime_environ::component::{TypeFuncIndex, MAX_FLAT_PARAMS};
use wasmtime_fiber::{Fiber, Suspend};
use wasmtime_runtime::{
    component::VMComponentContext, AsyncWasmCallState, SendSyncPtr, Store, VMFuncRef,
    VMOpaqueContext,
};

mod task_table;

const STATUS_NOT_STARTED: u32 = 0;
const STATUS_PARAMS_READ: u32 = 1;
const STATUS_RESULTS_WRITTEN: u32 = 2;
const STATUS_DONE: u32 = 3;

const EVENT_CALL_STARTED: u32 = 0;
const EVENT_CALL_RETURNED: u32 = 1;
const EVENT_CALL_DONE: u32 = 2;

const ENTER_FLAG_EXPECT_RETPTR: u32 = 1 << 0;
const EXIT_FLAG_ASYNC_CALLER: u32 = 1 << 0;
const EXIT_FLAG_ASYNC_CALLEE: u32 = 1 << 1;

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
    fiber: Option<StoreFiber>,
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

type Lower<T> = Box<
    dyn FnOnce(&mut StoreContextMut<T>, &mut [MaybeUninit<ValRaw>]) -> Result<()> + Send + Sync,
>;

type Lift<T> = Box<
    dyn FnOnce(&mut StoreContextMut<T>, &[ValRaw]) -> Result<Option<Box<dyn Any + Send + Sync>>>
        + Send
        + Sync,
>;

type LiftedResult = Box<dyn Any + Send + Sync>;

struct SendSyncConstPtr<T>(*const T);

impl<T> Clone for SendSyncConstPtr<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T> Copy for SendSyncConstPtr<T> {}

unsafe impl<T> Send for SendSyncConstPtr<T> {}
unsafe impl<T> Sync for SendSyncConstPtr<T> {}

pub struct ConcurrentState<T> {
    guest_task: Option<TaskId<T>>,
    futures: ReadyChunks<FuturesUnordered<HostTaskFuture<T>>>,
    table: TaskTable<T>,
    suspend: Option<
        SendSyncConstPtr<
            Suspend<Option<*mut dyn Store>, Option<*mut dyn Store>, (*mut dyn Store, Result<()>)>,
        >,
    >,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            guest_task: None,
            table: TaskTable::new(),
            futures: FuturesUnordered::new().ready_chunks(1024),
            suspend: None,
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
    store: &mut StoreContextMut<T>,
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

fn poll_loop<T>(store: &mut StoreContextMut<T>) -> Result<()> {
    let task = store.concurrent_state().guest_task;
    while task
        .map(|task| {
            Ok::<_, Error>(
                store
                    .concurrent_state()
                    .table
                    .get(task)?
                    .unwrap_guest_ref()
                    .result
                    .is_none(),
            )
        })
        .unwrap_or(Ok(true))?
    {
        let async_cx = store.0.async_cx().expect("async cx");
        let mut future = pin!(store.concurrent_state().futures.next());
        let ready = unsafe { async_cx.block_on(future.as_mut())? };

        if let Some(ready) = ready {
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

struct StoreFiber {
    fiber: Fiber<
        'static,
        Option<*mut dyn Store>,
        Option<*mut dyn Store>,
        (*mut dyn Store, Result<()>),
    >,
    state: Option<AsyncWasmCallState>,
}

unsafe impl Send for StoreFiber {}
unsafe impl Sync for StoreFiber {}

fn make_fiber<T>(
    store: &mut StoreContextMut<T>,
    fun: impl FnOnce(StoreContextMut<T>) -> Result<()> + 'static,
) -> Result<StoreFiber> {
    let stack = store.engine().allocator().allocate_fiber_stack()?;
    Ok(StoreFiber {
        fiber: Fiber::new(stack, move |store_ptr: Option<*mut dyn Store>, suspend| {
            let store_ptr = store_ptr.unwrap();
            let mut store = unsafe { StoreContextMut::from_raw(store_ptr) };
            let old_suspend = store
                .concurrent_state()
                .suspend
                .replace(SendSyncConstPtr(suspend));
            let result = fun(store.as_context_mut());
            store.concurrent_state().suspend = old_suspend;
            (store_ptr, result)
        })?,
        state: Some(AsyncWasmCallState::new()),
    })
}

fn resume_fiber<'a, T: 'static>(
    fiber: &mut StoreFiber,
    store: Option<StoreContextMut<'a, T>>,
) -> Result<(StoreContextMut<'a, T>, Result<()>), Option<StoreContextMut<'a, T>>> {
    unsafe {
        let state = fiber.state.take().unwrap().push();
        let result = fiber.fiber.resume(store.map(|s| s.0 as _));
        fiber.state = Some(state.restore());
        result
            .map(|(store, result)| (StoreContextMut::from_raw(store), result))
            .map_err(|v| v.map(|v| StoreContextMut::from_raw(v)))
    }
}

unsafe fn handle_result<T>(func: impl FnOnce() -> Result<T>) -> T {
    match panic::catch_unwind(AssertUnwindSafe(func)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => crate::trap::raise(e),
        Err(e) => wasmtime_runtime::resume_panic(e),
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

pub(crate) extern "C" fn async_enter<T>(
    cx: *mut VMOpaqueContext,
    start: *mut VMFuncRef,
    return_: *mut VMFuncRef,
    store_call: *mut VMFuncRef,
    params: u32,
    results: u32,
    call: u32,
    flags: u32,
) {
    unsafe {
        handle_result(|| {
            let expect_retptr = (flags & ENTER_FLAG_EXPECT_RETPTR) != 0;
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let start = SendSyncPtr::new(NonNull::new(start).unwrap());
            let return_ = SendSyncPtr::new(NonNull::new(return_).unwrap());
            let old_task = cx.concurrent_state().guest_task.take();
            let old_task_rep = old_task.map(|v| v.rep);
            let new_task = Task::Guest(GuestTask {
                lower_params: Some(Box::new(move |cx, dst| {
                    assert!(dst.len() <= MAX_FLAT_PARAMS);
                    let mut src = [MaybeUninit::uninit(); MAX_FLAT_PARAMS];
                    src[0] = MaybeUninit::new(ValRaw::u32(params));
                    let len = if expect_retptr {
                        src[1] = dst[0];
                        2
                    } else {
                        1
                    }
                    .max(dst.len());
                    crate::Func::call_unchecked_raw(
                        cx,
                        start.as_non_null(),
                        src.as_mut_ptr() as _,
                        len,
                    )?;
                    if !expect_retptr {
                        dst.copy_from_slice(&src[..dst.len()]);
                    }
                    let task = cx.concurrent_state().guest_task.unwrap();
                    if let Some(rep) = old_task_rep {
                        maybe_send_event(cx, TaskId::new(rep), EVENT_CALL_STARTED, task)?;
                    }
                    Ok(())
                })),
                lift_result: Some(Box::new(move |cx, src| {
                    let mut my_src = src.to_owned(); // TODO: use stack to avoid allocation?
                    my_src.push(ValRaw::u32(results));
                    crate::Func::call_unchecked_raw(
                        cx,
                        return_.as_non_null(),
                        my_src.as_mut_ptr(),
                        my_src.len(),
                    )?;
                    let task = cx.concurrent_state().guest_task.unwrap();
                    if let Some(rep) = old_task_rep {
                        maybe_send_event(cx, TaskId::new(rep), EVENT_CALL_RETURNED, task)?;
                    }
                    Ok(None)
                })),
                result: None,
                callback: None,
                caller: old_task.map(|task| Caller {
                    task,
                    store_call: SendSyncPtr::new(NonNull::new(store_call).unwrap()),
                    call,
                }),
                fiber: None,
            });
            let guest_task = if let Some(old_task) = old_task {
                cx.concurrent_state().table.push_child(new_task, old_task)
            } else {
                cx.concurrent_state().table.push(new_task)
            }?;
            cx.concurrent_state().guest_task = Some(guest_task);

            Ok(())
        })
    }
}

pub(crate) extern "C" fn async_exit<T: 'static>(
    cx: *mut VMOpaqueContext,
    callback: *mut VMFuncRef,
    guest_context: u32,
    callee: *mut VMFuncRef,
    param_count: u32,
    result_count: u32,
    flags: u32,
) -> u32 {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());

            let guest_task = cx.concurrent_state().guest_task.take().unwrap();

            let mut cx = if (flags & EXIT_FLAG_ASYNC_CALLEE) == 0 {
                let callee = SendSyncPtr::new(NonNull::new(callee).unwrap());
                let param_count = usize::try_from(param_count).unwrap();
                let result_count = usize::try_from(result_count).unwrap();
                assert!(param_count <= MAX_FLAT_PARAMS);
                assert!(result_count <= MAX_FLAT_PARAMS);

                // TODO: check if callee instance has already been entered and, if so, stash the callee in the task
                // so we can call it later.  Otherwise, mark it as entered, run the following code, and finally
                // mark it as unentered.

                let suspend = cx.concurrent_state().suspend.unwrap();

                let mut fiber = make_fiber(&mut cx, move |mut cx| {
                    let mut storage = [MaybeUninit::uninit(); MAX_FLAT_PARAMS];
                    let lower = cx
                        .concurrent_state()
                        .table
                        .get_mut(guest_task)?
                        .unwrap_guest_mut()
                        .lower_params
                        .take()
                        .unwrap();
                    lower(&mut cx, &mut storage[..param_count])?;

                    crate::Func::call_unchecked_raw(
                        &mut cx,
                        callee.as_non_null(),
                        storage.as_mut_ptr() as _,
                        param_count.max(result_count),
                    )?;

                    let lift = cx
                        .concurrent_state()
                        .table
                        .get_mut(guest_task)?
                        .unwrap_guest_mut()
                        .lift_result
                        .take()
                        .unwrap();

                    assert!(cx
                        .concurrent_state()
                        .table
                        .get(guest_task)?
                        .unwrap_guest_ref()
                        .result
                        .is_none());

                    let result = lift(
                        &mut cx,
                        mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(
                            &storage[..result_count],
                        ),
                    )?;
                    cx.concurrent_state()
                        .table
                        .get_mut(guest_task)?
                        .unwrap_guest_mut()
                        .result = result;

                    Ok(())
                })?;

                let mut cx = Some(cx);
                loop {
                    match resume_fiber(&mut fiber, cx.take()) {
                        Ok((cx, result)) => {
                            result?;
                            break cx;
                        }
                        Err(cx) => {
                            if let Some(mut cx) = cx {
                                cx.concurrent_state()
                                    .table
                                    .get_mut(guest_task)?
                                    .unwrap_guest_mut()
                                    .fiber = Some(fiber);
                                break cx;
                            } else {
                                (*suspend.0).suspend(None);
                            }
                        }
                    }
                }
            } else {
                cx
            };

            let caller = cx
                .concurrent_state()
                .table
                .get(guest_task)?
                .unwrap_guest_ref()
                .caller
                .as_ref()
                .map(|caller| (caller.task, caller.store_call.as_non_null(), caller.call));
            cx.concurrent_state().guest_task = caller.map(|(next, ..)| next);

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

                if (flags & EXIT_FLAG_ASYNC_CALLER) != 0 {
                    let (_, store_call, call) = caller.unwrap();
                    let mut src = [ValRaw::u32(call), ValRaw::u32(guest_task.rep)];
                    crate::Func::call_unchecked_raw(
                        &mut cx,
                        store_call,
                        src.as_mut_ptr(),
                        src.len(),
                    )?;
                } else {
                    poll_loop(&mut cx)?;
                }
            } else if status == STATUS_DONE {
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
        fiber: None,
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
            poll_loop(&mut store)?;
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
