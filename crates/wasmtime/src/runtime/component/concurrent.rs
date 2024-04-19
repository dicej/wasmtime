use crate::{AsContextMut, Engine, StoreContextMut, ValRaw};
use anyhow::{anyhow, Error, Result};
use futures::{
    future::{self, FutureExt},
    stream::{FuturesUnordered, ReadyChunks, StreamExt},
};
use once_cell::sync::Lazy;
use std::{
    any::Any,
    cell::UnsafeCell,
    future::Future,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    panic::{self, AssertUnwindSafe},
    pin::{pin, Pin},
    ptr::{self, NonNull},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
use task_table::TaskTable;
use wasmtime_environ::component::{TypeFuncIndex, MAX_FLAT_PARAMS};
use wasmtime_fiber::{Fiber, Suspend};
use wasmtime_runtime::{
    component::VMComponentContext,
    mpk::{self, ProtectionMask},
    AsyncWasmCallState, PreviousAsyncWasmCallState, SendSyncPtr, Store, VMFuncRef, VMOpaqueContext,
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
    dyn for<'a> FnOnce(
            StoreContextMut<'a, T>,
            &mut [MaybeUninit<ValRaw>],
        ) -> Result<StoreContextMut<'a, T>>
        + Send
        + Sync,
>;

type Lift<T> = Box<
    dyn for<'a> FnOnce(
            StoreContextMut<'a, T>,
            &[ValRaw],
        )
            -> Result<(Option<Box<dyn Any + Send + Sync>>, StoreContextMut<'a, T>)>
        + Send
        + Sync,
>;

type LiftedResult = Box<dyn Any + Send + Sync>;

struct Reset<T: Copy>(*mut T, T);

impl<T: Copy> Drop for Reset<T> {
    fn drop(&mut self) {
        unsafe {
            *self.0 = self.1;
        }
    }
}

struct AsyncState {
    current_suspend: UnsafeCell<
        *const Suspend<
            (Option<*mut dyn Store>, Result<()>),
            Option<*mut dyn Store>,
            (Option<*mut dyn Store>, Result<()>),
        >,
    >,
    current_poll_cx: UnsafeCell<*mut Context<'static>>,
}

unsafe impl Send for AsyncState {}
unsafe impl Sync for AsyncState {}

pub(crate) struct AsyncCx {
    current_suspend: *mut *const wasmtime_fiber::Suspend<
        (Option<*mut dyn Store>, Result<()>),
        Option<*mut dyn Store>,
        (Option<*mut dyn Store>, Result<()>),
    >,
    current_stack_limit: *mut usize,
    current_poll_cx: *mut *mut Context<'static>,
    track_pkey_context_switch: bool,
}

impl AsyncCx {
    pub(crate) fn new<T>(store: &mut StoreContextMut<T>) -> Self {
        Self {
            current_suspend: store.concurrent_state().async_state.current_suspend.get(),
            current_stack_limit: store.0.runtime_limits().stack_limit.get(),
            current_poll_cx: store.concurrent_state().async_state.current_poll_cx.get(),
            track_pkey_context_switch: store.has_pkey(),
        }
    }

    pub(crate) unsafe fn block_on<'a, T: 'static, U>(
        &self,
        mut future: Pin<&mut (dyn Future<Output = U> + Send)>,
        mut store: Option<StoreContextMut<'a, T>>,
    ) -> Result<(U, Option<StoreContextMut<'a, T>>)> {
        loop {
            let result = {
                let poll_cx = *self.current_poll_cx;
                let _reset = Reset(self.current_poll_cx, poll_cx);
                *self.current_poll_cx = ptr::null_mut();
                future.as_mut().poll(&mut *poll_cx)
            };

            match result {
                Poll::Ready(v) => break Ok((v, store)),
                Poll::Pending => {}
            }

            store = self.suspend(store)?;
        }
    }

    unsafe fn suspend<'a, T: 'static>(
        &self,
        store: Option<StoreContextMut<'a, T>>,
    ) -> Result<Option<StoreContextMut<'a, T>>> {
        let previous_mask = if self.track_pkey_context_switch {
            let previous_mask = mpk::current_mask();
            mpk::allow(ProtectionMask::all());
            previous_mask
        } else {
            ProtectionMask::all()
        };
        let store = suspend_fiber(self.current_suspend, self.current_stack_limit, store);
        if self.track_pkey_context_switch {
            mpk::allow(previous_mask);
        }
        store
    }
}

pub struct ConcurrentState<T> {
    guest_task: Option<TaskId<T>>,
    futures: ReadyChunks<FuturesUnordered<HostTaskFuture<T>>>,
    table: TaskTable<T>,
    async_state: AsyncState,
    result: Option<Box<dyn Any + Send + Sync>>,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            guest_task: None,
            table: TaskTable::new(),
            futures: FuturesUnordered::new().ready_chunks(1024),
            async_state: AsyncState {
                current_suspend: UnsafeCell::new(ptr::null()),
                current_poll_cx: UnsafeCell::new(ptr::null_mut()),
            },
            result: None,
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

pub(crate) fn poll_and_block<'a, T: 'static, R: Send + Sync + 'static>(
    mut store: StoreContextMut<'a, T>,
    future: impl Future<Output = impl FnOnce(StoreContextMut<T>) -> Result<R> + 'static>
        + Send
        + Sync
        + 'static,
) -> Result<(R, StoreContextMut<'a, T>)> {
    let async_cx = AsyncCx::new(&mut store);
    if let Some(caller) = store.concurrent_state().guest_task {
        let task = store
            .concurrent_state()
            .table
            .push_child(Task::Host(HostTask { caller }), caller)?;
        let mut future = Box::pin(future.map(move |fun| {
            (
                task.rep,
                Box::new(for_any(move |mut store| {
                    let result = fun(store.as_context_mut())?;
                    store
                        .concurrent_state()
                        .table
                        .get_mut(caller)?
                        .unwrap_guest_mut()
                        .result = Some(Box::new(result) as _);
                    Ok(())
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
                    fun(store.as_context_mut())?;
                    let result = *store
                        .concurrent_state()
                        .table
                        .get_mut(caller)?
                        .unwrap_guest_mut()
                        .result
                        .take()
                        .unwrap()
                        .downcast()
                        .unwrap();
                    (result, store)
                }
                Poll::Pending => {
                    store.concurrent_state().futures.get_mut().push(future);
                    loop {
                        if let Some(result) = store
                            .concurrent_state()
                            .table
                            .get_mut(caller)?
                            .unwrap_guest_mut()
                            .result
                            .take()
                        {
                            break (*result.downcast().unwrap(), store);
                        } else {
                            store = unsafe { async_cx.suspend(Some(store))?.unwrap() };
                        }
                    }
                }
            },
        )
    } else {
        let mut future = pin!(future);
        let (next, store) = unsafe { async_cx.block_on(future.as_mut(), Some(store))? };
        let mut store = store.unwrap();
        let ret = next(store.as_context_mut())?;
        Ok((ret, store))
    }
}

pub(crate) async fn on_fiber<'a, R: Send + Sync + 'static, T: 'static>(
    mut store: StoreContextMut<'a, T>,
    func: impl FnOnce(&mut StoreContextMut<T>) -> R + Send + 'static,
) -> Result<R> {
    let mut fiber = make_fiber(&mut store, move |mut store| {
        let result = func(&mut store);
        assert!(store.concurrent_state().result.is_none());
        store.concurrent_state().result = Some(Box::new(result) as _);
        Ok(())
    })?;

    let poll_cx = store.concurrent_state().async_state.current_poll_cx.get();
    future::poll_fn({
        let mut store = Some(store.as_context_mut());

        move |cx| unsafe {
            let _reset = Reset(poll_cx, *poll_cx);
            *poll_cx = mem::transmute::<&mut Context<'_>, *mut Context<'static>>(cx);
            match resume_fiber(&mut fiber, store.take(), Ok(())) {
                Ok((_, result)) => Poll::Ready(result),
                Err(s) => {
                    if let Some(range) = fiber.fiber.stack().range() {
                        AsyncWasmCallState::assert_current_state_not_in_range(range);
                    }
                    store = s;
                    Poll::Pending
                }
            }
        }
    })
    .await?;

    Ok(*store
        .concurrent_state()
        .result
        .take()
        .unwrap()
        .downcast()
        .unwrap())
}

fn maybe_send_event<'a, T: 'static>(
    mut store: StoreContextMut<'a, T>,
    guest_task: TaskId<T>,
    event: u32,
    call: TaskId<T>,
) -> Result<StoreContextMut<'a, T>> {
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
                &mut store,
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
                store = maybe_send_event(store, next, EVENT_CALL_DONE, guest_task)?;
            }
        }
        store.concurrent_state().guest_task = old_task;
    } else if let Some(mut fiber) = store
        .concurrent_state()
        .table
        .get_mut(guest_task)?
        .unwrap_guest_mut()
        .fiber
        .take()
    {
        match unsafe { resume_fiber(&mut fiber, Some(store), Ok(())) } {
            Ok((new_store, result)) => {
                result?;
                store = new_store;
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
                    store = maybe_send_event(store, next, EVENT_CALL_DONE, guest_task)?;
                }
            }
            Err(new_store) => {
                store = new_store.unwrap();
                store
                    .concurrent_state()
                    .table
                    .get_mut(guest_task)?
                    .unwrap_guest_mut()
                    .fiber = Some(fiber);
            }
        }
    }
    Ok(store)
}

fn poll_loop<'a, T: 'static>(mut store: StoreContextMut<'a, T>) -> Result<StoreContextMut<'a, T>> {
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
        let cx = AsyncCx::new(&mut store);
        let mut future = pin!(store.concurrent_state().futures.next());
        let (ready, _) = unsafe { cx.block_on::<T, _>(future.as_mut(), None)? };

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
                store = maybe_send_event(store, caller, EVENT_CALL_DONE, task)?;
            }
        } else {
            break;
        }
    }

    Ok(store)
}

struct StoreFiber {
    fiber: Fiber<
        'static,
        (Option<*mut dyn Store>, Result<()>),
        Option<*mut dyn Store>,
        (Option<*mut dyn Store>, Result<()>),
    >,
    state: Option<AsyncWasmCallState>,
    engine: Engine,
    suspend: *mut *const Suspend<
        (Option<*mut dyn Store>, Result<()>),
        Option<*mut dyn Store>,
        (Option<*mut dyn Store>, Result<()>),
    >,
    stack_limit: *mut usize,
}

impl Drop for StoreFiber {
    fn drop(&mut self) {
        if !self.fiber.done() {
            let result = unsafe { resume_fiber_raw(self, None, Err(anyhow!("future dropped"))) };
            debug_assert!(result.is_ok());
        }

        self.state.take().unwrap().assert_null();

        unsafe {
            self.engine
                .allocator()
                .deallocate_fiber_stack(self.fiber.stack());
        }
    }
}

unsafe impl Send for StoreFiber {}
unsafe impl Sync for StoreFiber {}

fn make_fiber<T>(
    store: &mut StoreContextMut<T>,
    fun: impl FnOnce(StoreContextMut<T>) -> Result<()> + 'static,
) -> Result<StoreFiber> {
    let engine = store.engine().clone();
    let stack = engine.allocator().allocate_fiber_stack()?;
    Ok(StoreFiber {
        fiber: Fiber::new(
            stack,
            move |(store_ptr, result): (Option<*mut dyn Store>, Result<()>), suspend| {
                if result.is_err() {
                    (store_ptr, result)
                } else {
                    unsafe {
                        let store_ptr = store_ptr.unwrap();
                        let mut store = StoreContextMut::from_raw(store_ptr);
                        let suspend_ptr =
                            store.concurrent_state().async_state.current_suspend.get();
                        let _reset = Reset(suspend_ptr, *suspend_ptr);
                        *suspend_ptr = suspend;
                        (Some(store_ptr), fun(store.as_context_mut()))
                    }
                }
            },
        )?,
        state: Some(AsyncWasmCallState::new()),
        engine,
        suspend: store.concurrent_state().async_state.current_suspend.get(),
        stack_limit: store.0.runtime_limits().stack_limit.get(),
    })
}

unsafe fn resume_fiber_raw(
    fiber: &mut StoreFiber,
    store: Option<*mut dyn Store>,
    result: Result<()>,
) -> Result<(Option<*mut dyn Store>, Result<()>), Option<*mut dyn Store>> {
    struct Restore<'a> {
        fiber: &'a mut StoreFiber,
        state: Option<PreviousAsyncWasmCallState>,
    }

    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            unsafe {
                self.fiber.state = Some(self.state.take().unwrap().restore());
            }
        }
    }

    unsafe {
        let _reset_suspend = Reset(fiber.suspend, *fiber.suspend);
        let _reset_stack_limit = Reset(fiber.stack_limit, *fiber.stack_limit);
        let state = Some(fiber.state.take().unwrap().push());
        let restore = Restore { fiber, state };
        restore.fiber.fiber.resume((store, result))
    }
}

unsafe fn resume_fiber<'a, T: 'static>(
    fiber: &mut StoreFiber,
    store: Option<StoreContextMut<'a, T>>,
    result: Result<()>,
) -> Result<(StoreContextMut<'a, T>, Result<()>), Option<StoreContextMut<'a, T>>> {
    resume_fiber_raw(fiber, store.map(|s| s.0 as _), result)
        .map(|(store, result)| (StoreContextMut::from_raw(store.unwrap()), result))
        .map_err(|v| v.map(|v| StoreContextMut::from_raw(v)))
}

unsafe fn suspend_fiber<'a, T: 'static>(
    suspend: *mut *const Suspend<
        (Option<*mut dyn Store>, Result<()>),
        Option<*mut dyn Store>,
        (Option<*mut dyn Store>, Result<()>),
    >,
    stack_limit: *mut usize,
    store: Option<StoreContextMut<'a, T>>,
) -> Result<Option<StoreContextMut<'a, T>>> {
    let _reset_suspend = Reset(suspend, *suspend);
    let _reset_stack_limit = Reset(stack_limit, *stack_limit);
    let (store, result) = (**suspend).suspend(store.map(|s| s.0 as _));
    result?;
    Ok(store.map(|v| StoreContextMut::from_raw(v)))
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
            lower(cx, storage)?;
            Ok(())
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

            let (result, mut cx) = lift(
                cx,
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

pub(crate) extern "C" fn async_enter<T: 'static>(
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
                lower_params: Some(Box::new(move |mut cx, dst| {
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
                        &mut cx,
                        start.as_non_null(),
                        src.as_mut_ptr() as _,
                        len,
                    )?;
                    if !expect_retptr {
                        dst.copy_from_slice(&src[..dst.len()]);
                    }
                    let task = cx.concurrent_state().guest_task.unwrap();
                    if let Some(rep) = old_task_rep {
                        cx = maybe_send_event(cx, TaskId::new(rep), EVENT_CALL_STARTED, task)?;
                    }
                    Ok(cx)
                })),
                lift_result: Some(Box::new(move |mut cx, src| {
                    let mut my_src = src.to_owned(); // TODO: use stack to avoid allocation?
                    my_src.push(ValRaw::u32(results));
                    crate::Func::call_unchecked_raw(
                        &mut cx,
                        return_.as_non_null(),
                        my_src.as_mut_ptr(),
                        my_src.len(),
                    )?;
                    let task = cx.concurrent_state().guest_task.unwrap();
                    if let Some(rep) = old_task_rep {
                        cx = maybe_send_event(cx, TaskId::new(rep), EVENT_CALL_RETURNED, task)?;
                    }
                    Ok((None, cx))
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

            let mut cx = if (flags & EXIT_FLAG_ASYNC_CALLEE) == 0 {
                let guest_task = cx.concurrent_state().guest_task.unwrap();
                let callee = SendSyncPtr::new(NonNull::new(callee).unwrap());
                let param_count = usize::try_from(param_count).unwrap();
                let result_count = usize::try_from(result_count).unwrap();
                assert!(param_count <= MAX_FLAT_PARAMS);
                assert!(result_count <= MAX_FLAT_PARAMS);

                // TODO: check if callee instance has already been entered and, if so, stash the callee in the task
                // so we can call it later.  Otherwise, mark it as entered, run the following code, and finally
                // mark it as unentered.

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
                    cx = lower(cx, &mut storage[..param_count])?;

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

                    let (result, mut cx) = lift(
                        cx,
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
                    match resume_fiber(&mut fiber, cx.take(), Ok(())) {
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
                                suspend_fiber::<T>(fiber.suspend, fiber.stack_limit, None)?;
                            }
                        }
                    }
                }
            } else {
                cx
            };

            let guest_task = cx.concurrent_state().guest_task.take().unwrap();

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
                    poll_loop(cx)?;
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

pub(crate) fn exit<'a, T: Send + 'static, R: 'static>(
    mut store: StoreContextMut<'a, T>,
    callback: NonNull<VMFuncRef>,
    guest_context: u32,
) -> Result<(R, StoreContextMut<'a, T>)> {
    if guest_context != 0 {
        let guest_task = store.concurrent_state().guest_task.unwrap();
        store
            .concurrent_state()
            .table
            .get_mut(guest_task)?
            .unwrap_guest_mut()
            .callback = Some((SendSyncPtr::new(callback), guest_context));

        {
            store = poll_loop(store)?;
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
        Ok((*result.downcast().unwrap(), store))
    } else {
        Err(anyhow!(crate::Trap::NoAsyncResult))
    }
}
