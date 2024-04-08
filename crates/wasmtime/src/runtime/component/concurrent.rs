use {
    crate::{
        component::func::{self, Func, Lower as _, LowerContext, Options},
        store::StoreOpaque,
        vm::{
            component::VMComponentContext,
            mpk::{self, ProtectionMask},
            AsyncWasmCallState, PreviousAsyncWasmCallState, SendSyncPtr, VMFuncRef,
            VMMemoryDefinition, VMOpaqueContext, VMStore,
        },
        AsContextMut, Engine, StoreContextMut, ValRaw,
    },
    anyhow::{anyhow, bail, Result},
    futures::{
        future::{self, Either, FutureExt},
        stream::{FuturesUnordered, StreamExt},
    },
    once_cell::sync::Lazy,
    ready_chunks::ReadyChunks,
    std::{
        any::Any,
        borrow::ToOwned,
        boxed::Box,
        cell::UnsafeCell,
        collections::{HashMap, VecDeque},
        future::Future,
        marker::PhantomData,
        mem::{self, MaybeUninit},
        panic::{self, AssertUnwindSafe},
        pin::{pin, Pin},
        ptr::{self, NonNull},
        sync::Arc,
        task::{Context, Poll, Wake, Waker},
        vec::Vec,
    },
    table::{Table, TableId},
    wasmtime_environ::component::{
        InterfaceType, RuntimeComponentInstanceIndex, StringEncoding, TypeFuncIndex,
        MAX_FLAT_PARAMS,
    },
    wasmtime_fiber::{Fiber, Suspend},
};

pub(crate) use futures_and_streams::{
    error_drop, flat_stream_receive, flat_stream_send, future_drop_receiver, future_drop_sender,
    future_new, future_receive, future_send, stream_drop_receiver, stream_drop_sender, stream_new,
    stream_receive, stream_send,
};
pub use futures_and_streams::{future, stream, Error, FutureReceiver, StreamReceiver};

mod futures_and_streams;
mod ready_chunks;
mod table;

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

type HostTaskFuture = Pin<
    Box<
        dyn Future<Output = (u32, Box<dyn FnOnce(*mut dyn VMStore) -> Result<()>>)>
            + Send
            + Sync
            + 'static,
    >,
>;

struct HostTask {
    caller: TableId<GuestTask>,
}

struct Caller {
    task: TableId<GuestTask>,
    store_call: SendSyncPtr<VMFuncRef>,
    call: u32,
}

struct GuestTask {
    lower_params: Option<RawLower>,
    lift_result: Option<RawLift>,
    result: Option<LiftedResult>,
    callback: Option<(SendSyncPtr<VMFuncRef>, u32)>,
    events: VecDeque<(u32, u32)>,
    caller: Option<Caller>,
    fiber: Option<StoreFiber>,
}

impl Default for GuestTask {
    fn default() -> Self {
        Self {
            lower_params: None,
            lift_result: None,
            result: None,
            callback: None,
            events: VecDeque::new(),
            caller: None,
            fiber: None,
        }
    }
}

type RawLower =
    Box<dyn FnOnce(*mut dyn VMStore, &mut [MaybeUninit<ValRaw>]) -> Result<()> + Send + Sync>;

type LowerFn<Context> = fn(Context, *mut dyn VMStore, &mut [MaybeUninit<ValRaw>]) -> Result<()>;

type RawLift = Box<
    dyn FnOnce(*mut dyn VMStore, &[ValRaw]) -> Result<Option<Box<dyn Any + Send + Sync>>>
        + Send
        + Sync,
>;

type LiftFn<Context> =
    fn(Context, *mut dyn VMStore, &[ValRaw]) -> Result<Option<Box<dyn Any + Send + Sync>>>;

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
        *mut Suspend<
            (Option<*mut dyn VMStore>, Result<()>),
            Option<*mut dyn VMStore>,
            (Option<*mut dyn VMStore>, Result<()>),
        >,
    >,
    current_poll_cx: UnsafeCell<*mut Context<'static>>,
}

unsafe impl Send for AsyncState {}
unsafe impl Sync for AsyncState {}

pub(crate) struct AsyncCx {
    current_suspend: *mut *mut wasmtime_fiber::Suspend<
        (Option<*mut dyn VMStore>, Result<()>),
        Option<*mut dyn VMStore>,
        (Option<*mut dyn VMStore>, Result<()>),
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

    unsafe fn poll<U>(&self, mut future: Pin<&mut (dyn Future<Output = U> + Send)>) -> Poll<U> {
        let poll_cx = *self.current_poll_cx;
        let _reset = Reset(self.current_poll_cx, poll_cx);
        *self.current_poll_cx = ptr::null_mut();
        assert!(!poll_cx.is_null());
        future.as_mut().poll(&mut *poll_cx)
    }

    pub(crate) unsafe fn block_on<'a, T, U>(
        &self,
        mut future: Pin<&mut (dyn Future<Output = U> + Send)>,
        mut store: Option<StoreContextMut<'a, T>>,
    ) -> Result<(U, Option<StoreContextMut<'a, T>>)> {
        loop {
            match self.poll(future.as_mut()) {
                Poll::Ready(v) => break Ok((v, store)),
                Poll::Pending => {}
            }

            store = self.suspend(store)?;
        }
    }

    unsafe fn suspend<'a, T>(
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
    guest_task: Option<TableId<GuestTask>>,
    futures: ReadyChunks<FuturesUnordered<HostTaskFuture>>,
    table: Table,
    async_state: AsyncState,
    sync_task_queues: HashMap<RuntimeComponentInstanceIndex, VecDeque<TableId<GuestTask>>>,
    _phantom: PhantomData<T>,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            guest_task: None,
            table: Table::new(),
            futures: ReadyChunks::new(FuturesUnordered::new(), 1024),
            async_state: AsyncState {
                current_suspend: UnsafeCell::new(ptr::null_mut()),
                current_poll_cx: UnsafeCell::new(ptr::null_mut()),
            },
            sync_task_queues: HashMap::new(),
            _phantom: PhantomData,
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

/// TODO: docs
pub fn for_any<F, R, T>(fun: F) -> F
where
    F: FnOnce(StoreContextMut<T>) -> R + 'static,
    R: 'static,
{
    fun
}

fn for_any_lower<
    F: FnOnce(*mut dyn VMStore, &mut [MaybeUninit<ValRaw>]) -> Result<()> + Send + Sync,
>(
    fun: F,
) -> F {
    fun
}

fn for_any_lift<
    F: FnOnce(*mut dyn VMStore, &[ValRaw]) -> Result<Option<Box<dyn Any + Send + Sync>>> + Send + Sync,
>(
    fun: F,
) -> F {
    fun
}

pub(crate) fn first_poll<T, R: Send + 'static>(
    mut store: StoreContextMut<T>,
    future: impl Future<Output = impl FnOnce(StoreContextMut<T>) -> Result<R> + 'static>
        + Send
        + Sync
        + 'static,
    lower: impl FnOnce(StoreContextMut<T>, R) -> Result<()> + Send + Sync + 'static,
) -> Result<Option<u32>> {
    let caller = store.concurrent_state().guest_task.unwrap();
    let task = store
        .concurrent_state()
        .table
        .push_child(HostTask { caller }, caller)?;
    let mut future = Box::pin(future.map(move |fun| {
        (
            task.rep(),
            Box::new(move |store| {
                let mut store = unsafe { StoreContextMut::from_raw(store) };
                let result = fun(store.as_context_mut())?;
                lower(store, result)
            }) as Box<dyn FnOnce(*mut dyn VMStore) -> Result<()>>,
        )
    })) as HostTaskFuture;

    Ok(
        match future
            .as_mut()
            .poll(&mut Context::from_waker(&dummy_waker()))
        {
            Poll::Ready((_, fun)) => {
                store.concurrent_state().table.delete(task)?;
                fun(store.0.traitobj())?;
                None
            }
            Poll::Pending => {
                store.concurrent_state().futures.get_mut().push(future);
                Some(task.rep())
            }
        },
    )
}

pub(crate) fn poll_and_block<'a, T, R: Send + Sync + 'static>(
    mut store: StoreContextMut<'a, T>,
    future: impl Future<Output = impl FnOnce(StoreContextMut<T>) -> Result<R> + 'static>
        + Send
        + Sync
        + 'static,
) -> Result<(R, StoreContextMut<'a, T>)> {
    let caller = store.concurrent_state().guest_task.unwrap();
    let old_result = store
        .concurrent_state()
        .table
        .get_mut(caller)?
        .result
        .take();
    let task = store
        .concurrent_state()
        .table
        .push_child(HostTask { caller }, caller)?;
    let mut future = Box::pin(future.map(move |fun| {
        (
            task.rep(),
            Box::new(move |store| {
                let mut store = unsafe { StoreContextMut::from_raw(store) };
                let result = fun(store.as_context_mut())?;
                store.concurrent_state().table.get_mut(caller)?.result =
                    Some(Box::new(result) as _);
                Ok(())
            }) as Box<dyn FnOnce(*mut dyn VMStore) -> Result<()>>,
        )
    })) as HostTaskFuture;

    Ok(
        match unsafe { AsyncCx::new(&mut store).poll(future.as_mut()) } {
            Poll::Ready((_, fun)) => {
                store.concurrent_state().table.delete(task)?;
                let store = store.0.traitobj();
                fun(store)?;
                let mut store = unsafe { StoreContextMut::from_raw(store) };
                let result = *mem::replace(
                    &mut store.concurrent_state().table.get_mut(caller)?.result,
                    old_result,
                )
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
                        .result
                        .take()
                    {
                        store.concurrent_state().table.get_mut(caller)?.result = old_result;
                        break (*result.downcast().unwrap(), store);
                    } else {
                        let async_cx = AsyncCx::new(&mut store);
                        store = unsafe { async_cx.suspend(Some(store))?.unwrap() };
                    }
                }
            }
        },
    )
}

pub(crate) async fn on_fiber<'a, R: Send + Sync + 'static, T: Send>(
    mut store: StoreContextMut<'a, T>,
    handle: Func,
    func: impl FnOnce(&mut StoreContextMut<T>) -> R + Send + 'static,
) -> Result<(R, StoreContextMut<'a, T>)> {
    assert!(store.concurrent_state().guest_task.is_none());

    let guest_task = store.concurrent_state().table.push(GuestTask::default())?;
    store.concurrent_state().guest_task = Some(guest_task);

    let is_concurrent = store.0[handle.0].options.async_();
    let instance = store.0[handle.0].component_instance;

    let mut fiber = make_fiber(&mut store, instance, move |mut store| {
        let result = func(&mut store);
        assert!(store
            .concurrent_state()
            .table
            .get(guest_task)?
            .result
            .is_none());
        store.concurrent_state().table.get_mut(guest_task)?.result = Some(Box::new(result) as _);
        Ok(())
    })?;

    if !is_concurrent {
        let queue = &mut store
            .concurrent_state()
            .sync_task_queues
            .entry(instance)
            .or_default();
        assert!(queue.is_empty());
        queue.push_back(guest_task);
    }

    #[derive(Clone, Copy)]
    struct PollCx(*mut *mut Context<'static>);

    unsafe impl Send for PollCx {}

    fn my_poll_fn<T, F: FnMut(&mut Context) -> Poll<T> + Send>(f: F) -> future::PollFn<F> {
        future::poll_fn(f)
    }

    let poll_cx = PollCx(store.concurrent_state().async_state.current_poll_cx.get());
    let mut store = my_poll_fn({
        let mut store = Some(store);

        move |cx| unsafe {
            if let Some(mut my_store) = store.take() {
                if let Poll::Ready(Some(ready)) =
                    my_store.concurrent_state().futures.poll_next_unpin(cx)
                {
                    match handle_ready(my_store, ready) {
                        Ok(s) => {
                            my_store = s;
                        }
                        Err(e) => {
                            return Poll::Ready(Err(e));
                        }
                    }
                }
                store = Some(my_store);
            }

            let _reset = Reset(poll_cx.0, *poll_cx.0);
            *poll_cx.0 = mem::transmute::<&mut Context<'_>, *mut Context<'static>>(cx);
            #[allow(dropping_copy_types)]
            drop(poll_cx);
            match resume_fiber(&mut fiber, store.take(), Ok(())) {
                Ok((store, result)) => Poll::Ready(result.map(|()| store)),
                Err(s) => {
                    if let Some(range) = fiber.fiber.as_ref().unwrap().stack().range() {
                        AsyncWasmCallState::assert_current_state_not_in_range(range);
                    }
                    store = s;
                    Poll::Pending
                }
            }
        }
    })
    .await?;

    let result = store
        .concurrent_state()
        .table
        .get_mut(guest_task)?
        .result
        .take();

    if !is_concurrent {
        store = future::poll_fn({
            let mut store = Some(store);

            move |cx| unsafe {
                let _reset = Reset(poll_cx.0, *poll_cx.0);
                *poll_cx.0 = mem::transmute::<&mut Context<'_>, *mut Context<'static>>(cx);
                #[allow(dropping_copy_types)]
                drop(poll_cx);
                Poll::Ready(resume_next_sync_task(
                    store.take().unwrap(),
                    guest_task,
                    instance,
                ))
            }
        })
        .await?;
    }

    if let Some(result) = result {
        if store
            .concurrent_state()
            .table
            .get(guest_task)?
            .callback
            .is_none()
        {
            // The task is finished -- delete it.
            //
            // Note that this isn't always the case -- a task may yield a result without completing, in which case
            // it should be polled until it's completed using `poll_for_completion`.
            store.concurrent_state().table.delete(guest_task)?;
            store.concurrent_state().guest_task = None;
        }
        Ok((*result.downcast().unwrap(), store))
    } else {
        // All outstanding host tasks completed, but the guest never yielded a result.
        Err(anyhow!(crate::Trap::NoAsyncResult))
    }
}

fn maybe_send_event<'a, T>(
    mut store: StoreContextMut<'a, T>,
    guest_task: TableId<GuestTask>,
    event: u32,
    call: u32,
) -> Result<StoreContextMut<'a, T>> {
    if let Some((callback, context)) = store.concurrent_state().table.get(guest_task)?.callback {
        let old_task = store.concurrent_state().guest_task.replace(guest_task);
        let params = &mut [
            ValRaw::u32(context),
            ValRaw::u32(event),
            ValRaw::u32(call),
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
            store.concurrent_state().table.get_mut(guest_task)?.callback = None;

            if let Some(next) = store
                .concurrent_state()
                .table
                .get(guest_task)?
                .caller
                .as_ref()
                .map(|c| c.task)
            {
                store.concurrent_state().table.delete(guest_task)?;
                store = maybe_send_event(store, next, EVENT_CALL_DONE, guest_task.rep())?;
            }
        }
        store.concurrent_state().guest_task = old_task;
        Ok(store)
    } else {
        store
            .concurrent_state()
            .table
            .get_mut(guest_task)?
            .events
            .push_back((event, call));

        if let Some(fiber) = store
            .concurrent_state()
            .table
            .get_mut(guest_task)?
            .fiber
            .take()
        {
            resume(store, guest_task, fiber)
        } else {
            Ok(store)
        }
    }
}

fn resume<'a, T>(
    mut store: StoreContextMut<'a, T>,
    guest_task: TableId<GuestTask>,
    mut fiber: StoreFiber,
) -> Result<StoreContextMut<'a, T>> {
    match unsafe { resume_fiber(&mut fiber, Some(store), Ok(())) } {
        Ok((mut store, result)) => {
            result?;
            store = resume_next_sync_task(store, guest_task, fiber.instance)?;
            if let Some(next) = store
                .concurrent_state()
                .table
                .get(guest_task)?
                .caller
                .as_ref()
                .map(|c| c.task)
            {
                store.concurrent_state().table.delete(guest_task)?;
                maybe_send_event(store, next, EVENT_CALL_DONE, guest_task.rep())
            } else {
                Ok(store)
            }
        }
        Err(new_store) => {
            store = new_store.unwrap();
            store.concurrent_state().table.get_mut(guest_task)?.fiber = Some(fiber);
            Ok(store)
        }
    }
}

fn poll_for_result<'a, T>(mut store: StoreContextMut<'a, T>) -> Result<StoreContextMut<'a, T>> {
    let task = store.concurrent_state().guest_task;
    poll_loop(store, move |store| {
        task.map(|task| {
            Ok::<_, anyhow::Error>(store.concurrent_state().table.get(task)?.result.is_none())
        })
        .unwrap_or(Ok(true))
    })
}

fn handle_ready<'a, T>(
    mut store: StoreContextMut<'a, T>,
    ready: Vec<(u32, Box<dyn FnOnce(*mut dyn VMStore) -> Result<()>>)>,
) -> Result<StoreContextMut<'a, T>> {
    for (task, fun) in ready {
        let task = TableId::<HostTask>::new(task);
        let vm_store = store.0.traitobj();
        fun(vm_store)?;
        store = unsafe { StoreContextMut::<T>::from_raw(vm_store) };
        let caller = store.concurrent_state().table.delete(task)?.caller;
        store = maybe_send_event(store, caller, EVENT_CALL_DONE, task.rep())?;
    }
    Ok(store)
}

fn poll_loop<'a, T>(
    mut store: StoreContextMut<'a, T>,
    continue_: impl Fn(&mut StoreContextMut<'a, T>) -> Result<bool>,
) -> Result<StoreContextMut<'a, T>> {
    while continue_(&mut store)? {
        let cx = AsyncCx::new(&mut store);
        let mut future = pin!(store.concurrent_state().futures.next());
        let (ready, _) = unsafe { cx.block_on::<T, _>(future.as_mut(), None)? };

        if let Some(ready) = ready {
            store = handle_ready(store, ready)?;
        } else {
            break;
        }
    }

    Ok(store)
}

fn resume_next_sync_task<'a, T>(
    mut store: StoreContextMut<'a, T>,
    current_task: TableId<GuestTask>,
    instance: RuntimeComponentInstanceIndex,
) -> Result<StoreContextMut<'a, T>> {
    assert_eq!(
        current_task.rep(),
        store
            .concurrent_state()
            .sync_task_queues
            .get_mut(&instance)
            .unwrap()
            .pop_front()
            .unwrap()
            .rep()
    );

    if let Some(next) = store
        .concurrent_state()
        .sync_task_queues
        .get_mut(&instance)
        .unwrap()
        .pop_front()
    {
        let fiber = store
            .concurrent_state()
            .table
            .get_mut(next)?
            .fiber
            .take()
            .unwrap();

        // TODO: Avoid tail calling `resume` here, because it may call us, leading to recursion limited only by
        // the number of waiters.  Flatten this into an iteration instead.
        resume(store, next, fiber)
    } else {
        store.concurrent_state().sync_task_queues.remove(&instance);
        Ok(store)
    }
}

struct StoreFiber {
    fiber: Option<
        Fiber<
            'static,
            (Option<*mut dyn VMStore>, Result<()>),
            Option<*mut dyn VMStore>,
            (Option<*mut dyn VMStore>, Result<()>),
        >,
    >,
    state: Option<AsyncWasmCallState>,
    engine: Engine,
    suspend: *mut *mut Suspend<
        (Option<*mut dyn VMStore>, Result<()>),
        Option<*mut dyn VMStore>,
        (Option<*mut dyn VMStore>, Result<()>),
    >,
    stack_limit: *mut usize,
    instance: RuntimeComponentInstanceIndex,
}

impl Drop for StoreFiber {
    fn drop(&mut self) {
        if !self.fiber.as_ref().unwrap().done() {
            let result = unsafe { resume_fiber_raw(self, None, Err(anyhow!("future dropped"))) };
            debug_assert!(result.is_ok());
        }

        self.state.take().unwrap().assert_null();

        unsafe {
            self.engine
                .allocator()
                .deallocate_fiber_stack(self.fiber.take().unwrap().into_stack());
        }
    }
}

unsafe impl Send for StoreFiber {}
unsafe impl Sync for StoreFiber {}

fn make_fiber<T>(
    store: &mut StoreContextMut<T>,
    instance: RuntimeComponentInstanceIndex,
    fun: impl FnOnce(StoreContextMut<T>) -> Result<()> + 'static,
) -> Result<StoreFiber> {
    let engine = store.engine().clone();
    let stack = engine.allocator().allocate_fiber_stack()?;
    Ok(StoreFiber {
        fiber: Some(Fiber::new(
            stack,
            move |(store_ptr, result): (Option<*mut dyn VMStore>, Result<()>), suspend| {
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
        )?),
        state: Some(AsyncWasmCallState::new()),
        engine,
        suspend: store.concurrent_state().async_state.current_suspend.get(),
        stack_limit: store.0.runtime_limits().stack_limit.get(),
        instance,
    })
}

unsafe fn resume_fiber_raw(
    fiber: &mut StoreFiber,
    store: Option<*mut dyn VMStore>,
    result: Result<()>,
) -> Result<(Option<*mut dyn VMStore>, Result<()>), Option<*mut dyn VMStore>> {
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
        restore
            .fiber
            .fiber
            .as_ref()
            .unwrap()
            .resume((store, result))
    }
}

unsafe fn resume_fiber<'a, T>(
    fiber: &mut StoreFiber,
    store: Option<StoreContextMut<'a, T>>,
    result: Result<()>,
) -> Result<(StoreContextMut<'a, T>, Result<()>), Option<StoreContextMut<'a, T>>> {
    resume_fiber_raw(fiber, store.map(|s| s.0.traitobj()), result)
        .map(|(store, result)| (StoreContextMut::from_raw(store.unwrap()), result))
        .map_err(|v| v.map(|v| StoreContextMut::from_raw(v)))
}

unsafe fn suspend_fiber<'a, T>(
    suspend: *mut *mut Suspend<
        (Option<*mut dyn VMStore>, Result<()>),
        Option<*mut dyn VMStore>,
        (Option<*mut dyn VMStore>, Result<()>),
    >,
    stack_limit: *mut usize,
    store: Option<StoreContextMut<'a, T>>,
) -> Result<Option<StoreContextMut<'a, T>>> {
    let _reset_suspend = Reset(suspend, *suspend);
    let _reset_stack_limit = Reset(stack_limit, *stack_limit);
    let (store, result) = (**suspend).suspend(store.map(|s| s.0.traitobj()));
    result?;
    Ok(store.map(|v| StoreContextMut::from_raw(v)))
}

unsafe fn handle_result<T>(func: impl FnOnce() -> Result<T>) -> T {
    match crate::runtime::vm::catch_unwind_and_longjmp(func) {
        Ok(value) => value,
        Err(e) => crate::trap::raise(e),
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
                .lower_params
                .take()
                .ok_or_else(|| anyhow!("call.start called more than once"))?;
            lower(cx.0.traitobj(), storage)?;
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
                .lift_result
                .take()
                .ok_or_else(|| anyhow!("call.return called more than once"))?;

            assert!(cx
                .concurrent_state()
                .table
                .get(guest_task)?
                .result
                .is_none());

            let cx = cx.0.traitobj();
            let result = lift(
                cx,
                mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(storage),
            )?;

            let mut cx = StoreContextMut::<T>::from_raw(cx);
            cx.concurrent_state().table.get_mut(guest_task)?.result = result;

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
            let old_task_rep = old_task.map(|v| v.rep());
            let new_task = GuestTask {
                lower_params: Some(Box::new(move |cx, dst| {
                    let mut cx = StoreContextMut::<T>::from_raw(cx);
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
                        maybe_send_event(cx, TableId::new(rep), EVENT_CALL_STARTED, task.rep())?;
                    }
                    Ok(())
                })),
                lift_result: Some(Box::new(move |cx, src| {
                    let mut cx = StoreContextMut::<T>::from_raw(cx);
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
                        maybe_send_event(cx, TableId::new(rep), EVENT_CALL_RETURNED, task.rep())?;
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
                events: VecDeque::new(),
            };
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

pub(crate) extern "C" fn async_exit<T>(
    cx: *mut VMOpaqueContext,
    callback: *mut VMFuncRef,
    guest_context: u32,
    callee: *mut VMFuncRef,
    callee_instance: RuntimeComponentInstanceIndex,
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

                let mut fiber = make_fiber(&mut cx, callee_instance, move |mut cx| {
                    let mut storage = [MaybeUninit::uninit(); MAX_FLAT_PARAMS];
                    let lower = cx
                        .concurrent_state()
                        .table
                        .get_mut(guest_task)?
                        .lower_params
                        .take()
                        .unwrap();
                    let cx = cx.0.traitobj();
                    lower(cx, &mut storage[..param_count])?;
                    let mut cx = StoreContextMut::<T>::from_raw(cx);

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
                        .lift_result
                        .take()
                        .unwrap();

                    assert!(cx
                        .concurrent_state()
                        .table
                        .get(guest_task)?
                        .result
                        .is_none());

                    let cx = cx.0.traitobj();
                    let result = lift(
                        cx,
                        mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(
                            &storage[..result_count],
                        ),
                    )?;
                    let mut cx = StoreContextMut::<T>::from_raw(cx);

                    cx.concurrent_state().table.get_mut(guest_task)?.result = result;

                    Ok(())
                })?;

                let queue = &mut cx
                    .concurrent_state()
                    .sync_task_queues
                    .entry(callee_instance)
                    .or_default();
                let first_in_queue = queue.is_empty();
                queue.push_back(guest_task);

                if first_in_queue {
                    let mut cx = Some(cx);
                    loop {
                        match resume_fiber(&mut fiber, cx.take(), Ok(())) {
                            Ok((cx, result)) => {
                                result?;
                                break resume_next_sync_task(cx, guest_task, callee_instance)?;
                            }
                            Err(cx) => {
                                if let Some(mut cx) = cx {
                                    cx.concurrent_state().table.get_mut(guest_task)?.fiber =
                                        Some(fiber);
                                    break cx;
                                } else {
                                    suspend_fiber::<T>(fiber.suspend, fiber.stack_limit, None)?;
                                }
                            }
                        }
                    }
                } else {
                    cx.concurrent_state().table.get_mut(guest_task)?.fiber = Some(fiber);
                    cx
                }
            } else {
                cx
            };

            let guest_task = cx.concurrent_state().guest_task.take().unwrap();

            let caller = cx
                .concurrent_state()
                .table
                .get(guest_task)?
                .caller
                .as_ref()
                .map(|caller| (caller.task, caller.store_call.as_non_null(), caller.call));
            cx.concurrent_state().guest_task = caller.map(|(next, ..)| next);

            let task = cx.concurrent_state().table.get_mut(guest_task)?;

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
                    let mut src = [ValRaw::u32(call), ValRaw::u32(guest_task.rep())];
                    crate::Func::call_unchecked_raw(
                        &mut cx,
                        store_call,
                        src.as_mut_ptr(),
                        src.len(),
                    )?;
                } else {
                    poll_for_result(cx)?;
                }
            } else if status == STATUS_DONE {
                cx.concurrent_state().table.delete(guest_task)?;
            }

            Ok(status)
        })
    }
}

pub(crate) extern "C" fn task_wait<T>(
    cx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    payload: u32,
) -> u32 {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(cx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());

            let guest_task = cx.concurrent_state().guest_task.unwrap();

            if cx
                .concurrent_state()
                .table
                .get(guest_task)?
                .callback
                .is_some()
            {
                bail!("cannot call `task.wait` from async-lifted export with callback");
            }

            let mut cx = poll_loop(cx, move |cx| {
                Ok::<_, anyhow::Error>(
                    cx.concurrent_state()
                        .table
                        .get(guest_task)?
                        .events
                        .is_empty(),
                )
            })?;

            let (event, call) = cx
                .concurrent_state()
                .table
                .get_mut(guest_task)?
                .events
                .pop_front()
                .unwrap();

            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                None,
                StringEncoding::Utf8,
                true,
                None,
            );
            let types = (*instance).component_types();
            let ptr = func::validate_inbounds::<(u32, u32)>(
                options.memory_mut(cx.0),
                &ValRaw::u32(payload),
            )?;
            let mut lower = LowerContext::new(cx, &options, types, instance);
            call.store(&mut lower, InterfaceType::U32, ptr)?;

            Ok(event)
        })
    }
}

pub(crate) fn enter<
    T: Send,
    LowerContext: Send + Sync + 'static,
    LiftContext: Send + Sync + 'static,
>(
    mut store: StoreContextMut<T>,
    lower_params: LowerFn<LowerContext>,
    lower_context: LowerContext,
    lift_result: LiftFn<LiftContext>,
    lift_context: LiftContext,
) -> Result<()> {
    let guest_task = store.concurrent_state().guest_task.unwrap();
    let task = store.concurrent_state().table.get_mut(guest_task)?;
    task.lower_params = Some(Box::new(for_any_lower(move |store, params| {
        lower_params(lower_context, store, params)
    })) as RawLower);
    task.lift_result = Some(Box::new(for_any_lift(move |store, result| {
        lift_result(lift_context, store, result)
    })) as RawLift);

    Ok(())
}

pub(crate) fn exit<'a, T: Send, R: 'static>(
    mut store: StoreContextMut<'a, T>,
    callback: NonNull<VMFuncRef>,
    guest_context: u32,
) -> Result<(R, StoreContextMut<'a, T>)> {
    let guest_task = store.concurrent_state().guest_task.unwrap();
    if guest_context != 0 {
        store.concurrent_state().table.get_mut(guest_task)?.callback =
            Some((SendSyncPtr::new(callback), guest_context));

        store = poll_for_result(store)?;
    }

    if let Some(result) = store
        .concurrent_state()
        .table
        .get_mut(guest_task)?
        .result
        .take()
    {
        Ok((*result.downcast().unwrap(), store))
    } else {
        // All outstanding host tasks completed, but the guest never yielded a result.
        Err(anyhow!(crate::Trap::NoAsyncResult))
    }
}

pub(crate) async fn poll<'a, T: Send>(
    mut store: StoreContextMut<'a, T>,
) -> Result<StoreContextMut<'a, T>> {
    let guest_task = store.concurrent_state().guest_task.unwrap();
    while store
        .concurrent_state()
        .table
        .get(guest_task)?
        .callback
        .is_some()
    {
        let ready = store.concurrent_state().futures.next().await.unwrap();
        store = handle_ready(store, ready)?;
    }

    Ok(store)
}

pub(crate) async fn poll_until<'a, T: Send, U>(
    mut store: StoreContextMut<'a, T>,
    future: impl Future<Output = U>,
) -> Result<(StoreContextMut<'a, T>, U)> {
    let mut future = Box::pin(future);
    loop {
        let ready = pin!(store.concurrent_state().futures.next());

        match future::select(ready, future).await {
            Either::Left((None, future_again)) => break Ok((store, future_again.await)),
            Either::Left((Some(ready), future_again)) => {
                store = handle_ready(store, ready)?;
                future = future_again;
            }
            Either::Right((result, _)) => break Ok((store, result)),
        }
    }
}
