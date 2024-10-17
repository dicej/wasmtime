use {
    super::{
        handle_result, table::TableId, HostTask, HostTaskFuture, STATUS_DONE, STATUS_NOT_STARTED,
    },
    crate::{
        component::{
            func::{self, Lift, LiftContext, Lower as _, LowerContext, Options},
            matching::InstanceType,
            Val,
        },
        vm::{
            component::{ComponentInstance, TableIndex, VMComponentContext},
            SendSyncPtr, VMFuncRef, VMMemoryDefinition, VMOpaqueContext, VMStore,
        },
        AsContextMut, StoreContextMut, ValRaw,
    },
    anyhow::{anyhow, bail, Result},
    futures::{channel::oneshot, future::FutureExt},
    std::{
        any::Any,
        boxed::Box,
        marker::PhantomData,
        mem::{self, MaybeUninit},
        ptr::NonNull,
        sync::Arc,
        vec::Vec,
    },
    wasmtime_environ::component::{
        CanonicalAbiInfo, ComponentTypes, InterfaceType, StringEncoding, TypeErrorTableIndex,
        TypeFutureTableIndex, TypeStreamTableIndex,
    },
};

// TODO: add `validate_inbounds` calls where appropriate

// TODO: Many of the functions in this module are used for both futures and streams, using runtime branches for
// specialization.  We should consider using generics instead to move those branches to compile time.

// TODO: Improve the host APIs for sending to and receiving from streams.  Currently, they require explicitly
// interleaving calls to `send` or `receive` and `StoreContextMut::wait_until`; see
// https://github.com/dicej/rfcs/blob/component-async/accepted/component-model-async.md#host-apis-for-creating-using-and-sharing-streams-futures-and-errors
// for an alternative approach.

fn receive_result(ty: TableIndex, types: &Arc<ComponentTypes>) -> InterfaceType {
    match ty {
        TableIndex::Future(ty) => InterfaceType::Result(types[ty].receive_result),
        TableIndex::Stream(ty) => InterfaceType::Option(types[ty].receive_option),
    }
}

fn send_result(ty: TableIndex, types: &Arc<ComponentTypes>) -> InterfaceType {
    InterfaceType::Result(match ty {
        TableIndex::Future(ty) => types[ty].send_result,
        TableIndex::Stream(ty) => types[ty].send_result,
    })
}

fn payload(ty: TableIndex, types: &Arc<ComponentTypes>) -> Option<InterfaceType> {
    match ty {
        TableIndex::Future(ty) => types[types[ty].ty].payload,
        TableIndex::Stream(ty) => Some(InterfaceType::List(types[types[ty].ty].list)),
    }
}

fn host_send<
    T: func::Lower + Send + Sync + 'static,
    W: func::Lower + Send + Sync + 'static,
    U,
    S: AsContextMut<Data = U>,
>(
    mut store: S,
    rep: u32,
    value: T,
    wrap: impl Fn(T) -> W + Send + Sync + 'static,
) -> Result<()> {
    let mut store = store.as_context_mut();
    let transmit = store
        .concurrent_state()
        .table
        .get(TableId::<TransmitSender>::new(rep))?
        .0;
    let transmit = store.concurrent_state().table.get_mut(transmit)?;
    let new_state = if let ReceiveState::Closed = &transmit.receive {
        ReceiveState::Closed
    } else {
        ReceiveState::Open
    };

    match mem::replace(&mut transmit.receive, new_state) {
        ReceiveState::Open => {
            assert!(matches!(&transmit.send, SendState::Open));

            transmit.send = SendState::HostReady {
                accept: Box::new(move |receiver| {
                    match receiver {
                        Receiver::Guest {
                            lower:
                                RawLowerContext {
                                    store,
                                    options,
                                    types,
                                    instance,
                                },
                            ty,
                            offset,
                        } => {
                            let lower = &mut unsafe {
                                LowerContext::new(
                                    StoreContextMut::<U>::from_raw(store),
                                    options,
                                    types,
                                    instance,
                                )
                            };
                            wrap(value).store(lower, ty, offset)?;
                        }
                        Receiver::Host { accept } => accept(Box::new(value))?,
                        Receiver::None => {}
                    }
                    Ok(())
                }),
                close: false,
            };
        }

        ReceiveState::GuestReady {
            ty,
            flat_abi: _,
            options,
            results,
            instance,
            tx: _receive_tx,
            entry,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let lower =
                &mut LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());
            wrap(value).store(
                lower,
                receive_result(ty, types),
                usize::try_from(results).unwrap(),
            )?;

            if let Some(entry) = entry {
                (*instance.as_ptr()).handle_table().insert(entry);
            }
        },

        ReceiveState::HostReady { accept } => {
            accept(Sender::Host {
                value: Box::new(value),
            })?;
        }

        // TODO: should we return an error here?
        ReceiveState::Closed => {}
    }

    Ok(())
}

pub fn host_receive<T: func::Lift + Sync + Send + 'static, U, S: AsContextMut<Data = U>>(
    mut store: S,
    rep: u32,
) -> Result<oneshot::Receiver<Option<T>>> {
    let mut store = store.as_context_mut();
    let (tx, rx) = oneshot::channel();
    let transmit_id = store
        .concurrent_state()
        .table
        .get(TableId::<TransmitReceiver>::new(rep))?
        .0;
    let transmit = store.concurrent_state().table.get_mut(transmit_id)?;
    let new_state = if let SendState::Closed = &transmit.send {
        SendState::Closed
    } else {
        SendState::Open
    };

    match mem::replace(&mut transmit.send, new_state) {
        SendState::Open => {
            assert!(matches!(&transmit.receive, ReceiveState::Open));

            transmit.receive = ReceiveState::HostReady {
                accept: Box::new(move |sender| {
                    match sender {
                        Sender::Guest { lift, ty, ptr } => {
                            _ = tx.send(
                                ty.map(|ty| {
                                    T::load(
                                        lift,
                                        ty,
                                        &lift.memory()[usize::try_from(ptr).unwrap()..]
                                            [..T::SIZE32],
                                    )
                                })
                                .transpose()?,
                            );
                        }
                        Sender::Host { value } => {
                            _ = tx.send(Some(
                                *value
                                    .downcast()
                                    .map_err(|_| anyhow!("transmit type mismatch"))?,
                            ));
                        }
                        Sender::None => {}
                    }
                    Ok(())
                }),
            };
        }

        SendState::GuestReady {
            ty,
            flat_abi: _,
            options,
            params,
            results,
            instance,
            tx: _send_tx,
            entry,
            close,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let lift = &mut LiftContext::new(store.0, &options, types, instance.as_ptr());
            _ = tx.send(
                payload(ty, types)
                    .map(|ty| {
                        T::load(
                            lift,
                            ty,
                            &lift.memory()[usize::try_from(params).unwrap()..][..T::SIZE32],
                        )
                    })
                    .transpose()?,
            );

            let mut lower =
                LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());
            Ok::<(), Error>(()).store(
                &mut lower,
                send_result(ty, types),
                usize::try_from(results).unwrap(),
            )?;

            if close {
                store.concurrent_state().table.get_mut(transmit_id)?.send = SendState::Closed;
            } else if let Some(entry) = entry {
                (*instance.as_ptr()).handle_table().insert(entry);
            }
        },

        SendState::HostReady { accept, close } => {
            accept(Receiver::Host {
                accept: Box::new(move |any| {
                    _ = tx.send(Some(
                        *any.downcast()
                            .map_err(|_| anyhow!("transmit type mismatch"))?,
                    ));
                    Ok(())
                }),
            })?;

            if close {
                store.concurrent_state().table.get_mut(transmit_id)?.send = SendState::Closed;
            }
        }

        SendState::Closed => {}
    }

    Ok(rx)
}

fn host_close_sender<U, S: AsContextMut<Data = U>>(mut store: S, rep: u32) -> Result<()> {
    let mut store = store.as_context_mut();
    let sender = store
        .concurrent_state()
        .table
        .delete::<TransmitSender>(TableId::new(rep))?;
    let transmit = store.concurrent_state().table.get_mut(sender.0)?;

    match &mut transmit.send {
        SendState::GuestReady { close, .. } => {
            *close = true;
        }

        SendState::HostReady { close, .. } => {
            *close = true;
        }

        v @ SendState::Open => {
            *v = SendState::Closed;
        }

        SendState::Closed => unreachable!(),
    }

    let new_state = if let ReceiveState::Closed = &transmit.receive {
        ReceiveState::Closed
    } else {
        ReceiveState::Open
    };

    match mem::replace(&mut transmit.receive, new_state) {
        ReceiveState::GuestReady {
            ty,
            flat_abi: _,
            options,
            results,
            instance,
            tx: _tx,
            entry,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let mut lower =
                LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());

            match ty {
                TableIndex::Future(ty) => {
                    Err::<(), _>(Error { rep: 0 }).store(
                        &mut lower,
                        InterfaceType::Result(types[ty].receive_result),
                        usize::try_from(results).unwrap(),
                    )?;
                }
                TableIndex::Stream(ty) => {
                    Val::Option(None).store(
                        &mut lower,
                        InterfaceType::Option(types[ty].receive_option),
                        usize::try_from(results).unwrap(),
                    )?;
                }
            }

            if let Some(entry) = entry {
                (*instance.as_ptr()).handle_table().insert(entry);
            }
        },

        ReceiveState::HostReady { accept } => {
            accept(Sender::None)?;
        }

        ReceiveState::Open => {}

        ReceiveState::Closed => {
            store.concurrent_state().table.delete(sender.0)?;
        }
    }
    Ok(())
}

fn host_close_receiver<U, S: AsContextMut<Data = U>>(mut store: S, rep: u32) -> Result<()> {
    let mut store = store.as_context_mut();
    let receiver = store
        .concurrent_state()
        .table
        .delete::<TransmitReceiver>(TableId::new(rep))?;
    let transmit = store.concurrent_state().table.get_mut(receiver.0)?;

    transmit.receive = ReceiveState::Closed;

    let new_state = if let SendState::Closed = &transmit.send {
        SendState::Closed
    } else {
        SendState::Open
    };

    match mem::replace(&mut transmit.send, new_state) {
        SendState::GuestReady {
            ty,
            flat_abi: _,
            options,
            params: _,
            instance,
            results,
            tx: _tx,
            entry,
            close,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let mut lower =
                LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());
            Err::<(), _>(Error { rep: 0 }).store(
                &mut lower,
                send_result(ty, types),
                usize::try_from(results).unwrap(),
            )?;

            if close {
                store.concurrent_state().table.delete(receiver.0)?;
            } else if let Some(entry) = entry {
                (*instance.as_ptr()).handle_table().insert(entry);
            }
        },

        SendState::HostReady { accept, close } => {
            accept(Receiver::None)?;

            if close {
                store.concurrent_state().table.delete(receiver.0)?;
            }
        }

        SendState::Open => {}

        SendState::Closed => {
            store.concurrent_state().table.delete(receiver.0)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlatAbi {
    size: u32,
    align: u32,
}

/// TODO: docs
pub struct FutureSender<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> FutureSender<T> {
    /// TODO: docs
    pub fn send<U, S: AsContextMut<Data = U>>(self, store: S, value: T) -> Result<()>
    where
        T: func::Lower + Send + Sync + 'static,
    {
        host_send(store, self.rep, value, |v| Ok::<_, Error>(v))
    }

    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        host_close_sender(store, self.rep)
    }
}

/// TODO: docs
pub struct FutureReceiver<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> FutureReceiver<T> {
    /// TODO: docs
    pub fn receive<U, S: AsContextMut<Data = U>>(
        self,
        store: S,
    ) -> Result<oneshot::Receiver<Option<T>>>
    where
        T: func::Lift + Sync + Send + 'static,
    {
        host_receive(store, self.rep)
    }

    fn lower_to_index<U>(&self, cx: &mut LowerContext<'_, U>, ty: InterfaceType) -> Result<u32> {
        match ty {
            InterfaceType::Future(dst) => {
                unsafe {
                    assert!((*cx.instance)
                        .handle_table()
                        .insert((TableIndex::Future(dst), self.rep)));
                }

                Ok(self.rep)
            }
            _ => func::bad_type_info(),
        }
    }

    fn lift_from_index(cx: &mut LiftContext<'_>, ty: InterfaceType, index: u32) -> Result<Self> {
        match ty {
            InterfaceType::Future(src) => {
                unsafe {
                    if !(*cx.instance)
                        .handle_table()
                        .remove(&(TableIndex::Future(src), index))
                    {
                        bail!("invalid handle");
                    }
                }

                Ok(Self {
                    rep: index,
                    _phantom: PhantomData,
                })
            }
            _ => func::bad_type_info(),
        }
    }

    /// TODO: docs
    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        host_close_receiver(store, self.rep)
    }
}

unsafe impl<T> func::ComponentType for FutureReceiver<T> {
    const ABI: CanonicalAbiInfo = CanonicalAbiInfo::SCALAR4;

    type Lower = <u32 as func::ComponentType>::Lower;

    fn typecheck(ty: &InterfaceType, _types: &InstanceType<'_>) -> Result<()> {
        match ty {
            InterfaceType::Future(_) => Ok(()),
            other => bail!("expected `future`, found `{}`", func::desc(other)),
        }
    }
}

unsafe impl<T> func::Lower for FutureReceiver<T> {
    fn lower<U>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
        dst: &mut MaybeUninit<Self::Lower>,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .lower(cx, InterfaceType::U32, dst)
    }

    fn store<U>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
        offset: usize,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .store(cx, InterfaceType::U32, offset)
    }
}

unsafe impl<T> func::Lift for FutureReceiver<T> {
    fn lift(cx: &mut LiftContext<'_>, ty: InterfaceType, src: &Self::Lower) -> Result<Self> {
        let index = u32::lift(cx, InterfaceType::U32, src)?;
        Self::lift_from_index(cx, ty, index)
    }

    fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Self> {
        let index = u32::load(cx, InterfaceType::U32, bytes)?;
        Self::lift_from_index(cx, ty, index)
    }
}

/// TODO: docs
pub fn future<T, U, S: AsContextMut<Data = U>>(
    mut store: S,
) -> Result<(FutureSender<T>, FutureReceiver<T>)> {
    let mut store = store.as_context_mut();
    let transmit = store.concurrent_state().table.push(TransmitState {
        receive: ReceiveState::Open,
        send: SendState::Open,
    })?;
    let sender = store
        .concurrent_state()
        .table
        .push_child(TransmitSender(transmit), transmit)?;
    let receiver = store
        .concurrent_state()
        .table
        .push_child(TransmitReceiver(transmit), transmit)?;

    Ok((
        FutureSender {
            rep: sender.rep(),
            _phantom: PhantomData,
        },
        FutureReceiver {
            rep: receiver.rep(),
            _phantom: PhantomData,
        },
    ))
}

/// TODO: docs
pub struct StreamSender<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> StreamSender<T> {
    /// TODO: docs
    pub fn send<U, S: AsContextMut<Data = U>>(&mut self, store: S, values: Vec<T>) -> Result<()>
    where
        T: func::Lower + Send + Sync + 'static,
    {
        host_send(store, self.rep, values, |v| Some(Ok::<_, Error>(v)))
    }

    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        host_close_sender(store, self.rep)
    }
}

/// TODO: docs
pub struct StreamReceiver<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> StreamReceiver<T> {
    /// TODO: docs
    pub fn receive<U, S: AsContextMut<Data = U>>(
        &mut self,
        store: S,
    ) -> Result<oneshot::Receiver<Option<Vec<T>>>>
    where
        T: func::Lift + Sync + Send + 'static,
    {
        host_receive(store, self.rep)
    }

    fn lower_to_index<U>(&self, cx: &mut LowerContext<'_, U>, ty: InterfaceType) -> Result<u32> {
        match ty {
            InterfaceType::Stream(dst) => {
                unsafe {
                    assert!((*cx.instance)
                        .handle_table()
                        .insert((TableIndex::Stream(dst), self.rep)));
                }

                Ok(self.rep)
            }
            _ => func::bad_type_info(),
        }
    }

    fn lift_from_index(cx: &mut LiftContext<'_>, ty: InterfaceType, index: u32) -> Result<Self> {
        match ty {
            InterfaceType::Stream(src) => {
                unsafe {
                    if !(*cx.instance)
                        .handle_table()
                        .remove(&(TableIndex::Stream(src), index))
                    {
                        bail!("invalid handle");
                    }
                }

                Ok(Self {
                    rep: index,
                    _phantom: PhantomData,
                })
            }
            _ => func::bad_type_info(),
        }
    }

    /// TODO: docs
    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        host_close_receiver(store, self.rep)
    }
}

unsafe impl<T> func::ComponentType for StreamReceiver<T> {
    const ABI: CanonicalAbiInfo = CanonicalAbiInfo::SCALAR4;

    type Lower = <u32 as func::ComponentType>::Lower;

    fn typecheck(ty: &InterfaceType, _types: &InstanceType<'_>) -> Result<()> {
        match ty {
            InterfaceType::Stream(_) => Ok(()),
            other => bail!("expected `stream`, found `{}`", func::desc(other)),
        }
    }
}

unsafe impl<T> func::Lower for StreamReceiver<T> {
    fn lower<U>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
        dst: &mut MaybeUninit<Self::Lower>,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .lower(cx, InterfaceType::U32, dst)
    }

    fn store<U>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
        offset: usize,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .store(cx, InterfaceType::U32, offset)
    }
}

unsafe impl<T> func::Lift for StreamReceiver<T> {
    fn lift(cx: &mut LiftContext<'_>, ty: InterfaceType, src: &Self::Lower) -> Result<Self> {
        let index = u32::lift(cx, InterfaceType::U32, src)?;
        Self::lift_from_index(cx, ty, index)
    }

    fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Self> {
        let index = u32::load(cx, InterfaceType::U32, bytes)?;
        Self::lift_from_index(cx, ty, index)
    }
}

/// TODO: docs
pub fn stream<T, U, S: AsContextMut<Data = U>>(
    mut store: S,
) -> Result<(StreamSender<T>, StreamReceiver<T>)> {
    let mut store = store.as_context_mut();
    let transmit = store.concurrent_state().table.push(TransmitState {
        receive: ReceiveState::Open,
        send: SendState::Open,
    })?;
    let sender = store
        .concurrent_state()
        .table
        .push_child(TransmitSender(transmit), transmit)?;
    let receiver = store
        .concurrent_state()
        .table
        .push_child(TransmitReceiver(transmit), transmit)?;

    Ok((
        StreamSender {
            rep: sender.rep(),
            _phantom: PhantomData,
        },
        StreamReceiver {
            rep: receiver.rep(),
            _phantom: PhantomData,
        },
    ))
}

/// TODO: docs
pub struct Error {
    rep: u32,
}

impl Error {
    fn lower_to_index<U>(&self, cx: &mut LowerContext<'_, U>, ty: InterfaceType) -> Result<u32> {
        match ty {
            InterfaceType::Error(dst) => {
                unsafe {
                    *(*cx.instance)
                        .error_table()
                        .entry((dst, self.rep))
                        .or_default() += 1;
                }
                Ok(self.rep)
            }
            _ => func::bad_type_info(),
        }
    }

    fn lift_from_index(cx: &mut LiftContext<'_>, ty: InterfaceType, index: u32) -> Result<Self> {
        match ty {
            InterfaceType::Error(src) => {
                if !unsafe { (*cx.instance).error_table().contains_key(&(src, index)) } {
                    bail!("invalid handle");
                }
                Ok(Self { rep: index })
            }
            _ => func::bad_type_info(),
        }
    }
}

unsafe impl func::ComponentType for Error {
    const ABI: CanonicalAbiInfo = CanonicalAbiInfo::SCALAR4;

    type Lower = <u32 as func::ComponentType>::Lower;

    fn typecheck(ty: &InterfaceType, _types: &InstanceType<'_>) -> Result<()> {
        match ty {
            InterfaceType::Error(_) => Ok(()),
            other => bail!("expected `error`, found `{}`", func::desc(other)),
        }
    }
}

unsafe impl func::Lower for Error {
    fn lower<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        dst: &mut MaybeUninit<Self::Lower>,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .lower(cx, InterfaceType::U32, dst)
    }

    fn store<T>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        offset: usize,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .store(cx, InterfaceType::U32, offset)
    }
}

unsafe impl func::Lift for Error {
    fn lift(cx: &mut LiftContext<'_>, ty: InterfaceType, src: &Self::Lower) -> Result<Self> {
        let index = u32::lift(cx, InterfaceType::U32, src)?;
        Self::lift_from_index(cx, ty, index)
    }

    fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Self> {
        let index = u32::load(cx, InterfaceType::U32, bytes)?;
        Self::lift_from_index(cx, ty, index)
    }
}

struct TransmitState {
    send: SendState,
    receive: ReceiveState,
}

struct TransmitSender(TableId<TransmitState>);

impl Clone for TransmitSender {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Copy for TransmitSender {}

struct TransmitReceiver(TableId<TransmitState>);

impl Clone for TransmitReceiver {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Copy for TransmitReceiver {}

enum SendState {
    Open,
    GuestReady {
        ty: TableIndex,
        flat_abi: Option<FlatAbi>,
        options: Options,
        params: u32,
        results: u32,
        instance: SendSyncPtr<ComponentInstance>,
        tx: oneshot::Sender<()>,
        entry: Option<(TableIndex, u32)>,
        close: bool,
    },
    HostReady {
        accept: Box<dyn FnOnce(Receiver) -> Result<()> + Send + Sync>,
        close: bool,
    },
    Closed,
}

enum ReceiveState {
    Open,
    GuestReady {
        ty: TableIndex,
        flat_abi: Option<FlatAbi>,
        options: Options,
        results: u32,
        instance: SendSyncPtr<ComponentInstance>,
        tx: oneshot::Sender<()>,
        entry: Option<(TableIndex, u32)>,
    },
    HostReady {
        accept: Box<dyn FnOnce(Sender) -> Result<()> + Send + Sync>,
    },
    Closed,
}

enum Sender<'a> {
    Guest {
        lift: &'a mut LiftContext<'a>,
        ty: Option<InterfaceType>,
        ptr: u32,
    },
    Host {
        value: Box<dyn Any>,
    },
    None,
}

struct RawLowerContext<'a> {
    store: *mut dyn VMStore,
    options: &'a Options,
    types: &'a ComponentTypes,
    instance: *mut ComponentInstance,
}

enum Receiver<'a> {
    Guest {
        lower: RawLowerContext<'a>,
        ty: InterfaceType,
        offset: usize,
    },
    Host {
        accept: Box<dyn FnOnce(Box<dyn Any>) -> Result<()>>,
    },
    None,
}

fn guest_new<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    ty: TableIndex,
    results: u32,
) {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                None,
                StringEncoding::Utf8,
                true,
                None,
            );
            let types = (*instance).component_types();
            let transmit = cx.concurrent_state().table.push(TransmitState {
                receive: ReceiveState::Open,
                send: SendState::Open,
            })?;
            let sender = cx
                .concurrent_state()
                .table
                .push_child(TransmitSender(transmit), transmit)?;
            let receiver = cx
                .concurrent_state()
                .table
                .push_child(TransmitReceiver(transmit), transmit)?;
            (*instance).handle_table().insert((ty, sender.rep()));
            (*instance).handle_table().insert((ty, receiver.rep()));
            let ptr = func::validate_inbounds::<(u32, u32)>(
                options.memory_mut(cx.0),
                &ValRaw::u32(results),
            )?;
            let mut lower = LowerContext::new(cx, &options, types, instance);
            sender.rep().store(&mut lower, InterfaceType::U32, ptr)?;
            receiver
                .rep()
                .store(&mut lower, InterfaceType::U32, ptr + 4)?;
            Ok(())
        })
    }
}

unsafe fn copy<T>(
    mut cx: StoreContextMut<'_, T>,
    types: &Arc<ComponentTypes>,
    instance: *mut ComponentInstance,
    flat_abi: Option<FlatAbi>,
    send_ty: TableIndex,
    send_options: &Options,
    send_params: u32,
    receive_ty: TableIndex,
    receive_options: &Options,
    receive_results: u32,
) -> Result<()> {
    match (send_ty, receive_ty) {
        (TableIndex::Future(send_ty), TableIndex::Future(receive_ty)) => {
            let val = types[types[send_ty].ty]
                .payload
                .map(|ty| {
                    let lift = &mut LiftContext::new(cx.0, send_options, types, instance);
                    Val::load(
                        lift,
                        ty,
                        &lift.memory()[usize::try_from(send_params).unwrap()..]
                            [..usize::try_from(types.canonical_abi(&ty).size32).unwrap()],
                    )
                })
                .transpose()?;

            let mut lower =
                LowerContext::new(cx.as_context_mut(), receive_options, types, instance);
            Val::Result(Ok(val.map(Box::new))).store(
                &mut lower,
                InterfaceType::Result(types[receive_ty].receive_result),
                usize::try_from(receive_results).unwrap(),
            )?;
        }
        (TableIndex::Stream(send_ty), TableIndex::Stream(receive_ty)) => {
            let lift = &mut LiftContext::new(cx.0, send_options, types, instance);
            let list = InterfaceType::List(types[types[send_ty].ty].list);
            let src = &lift.memory()[usize::try_from(send_params).unwrap()..][..8];
            if let Some(flat_abi) = flat_abi {
                // Fast path memcpy for "flat" (i.e. no pointers or handles) payloads:
                let length = u32::load(lift, InterfaceType::U32, &src[4..][..4])?;
                let src = usize::try_from(u32::load(lift, InterfaceType::U32, &src[..4])?).unwrap();

                let mut lower =
                    LowerContext::new(cx.as_context_mut(), receive_options, types, instance);

                let length_in_bytes = usize::try_from(flat_abi.size * length).unwrap();

                let dst = lower.realloc(0, 0, flat_abi.align, length_in_bytes)?;

                {
                    let src = send_options.memory(cx.0)[src..][..length_in_bytes].as_ptr();
                    let dst =
                        receive_options.memory_mut(cx.0)[dst..][..length_in_bytes].as_mut_ptr();
                    src.copy_to(dst, length_in_bytes);
                }

                let mut lower =
                    LowerContext::new(cx.as_context_mut(), receive_options, types, instance);

                let offset = usize::try_from(receive_results).unwrap();
                // `Some` discriminant:
                1u32.store(&mut lower, InterfaceType::U32, offset)?;
                // `Ok` discriminant:
                0u32.store(&mut lower, InterfaceType::U32, offset + 4)?;
                // List pointer:
                u32::try_from(dst)
                    .unwrap()
                    .store(&mut lower, InterfaceType::U32, offset + 8)?;
                // List length:
                length.store(&mut lower, InterfaceType::U32, offset + 12)?;
            } else {
                let val = Val::load(lift, list, src)?;

                let mut lower =
                    LowerContext::new(cx.as_context_mut(), receive_options, types, instance);
                Val::Option(Some(Box::new(Val::Result(Ok(Some(Box::new(val))))))).store(
                    &mut lower,
                    InterfaceType::Option(types[receive_ty].receive_option),
                    usize::try_from(receive_results).unwrap(),
                )?;
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}

fn guest_send<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TableIndex,
    flat_abi: Option<FlatAbi>,
    mut params: u32,
    results: u32,
) -> u32 {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                NonNull::new(realloc),
                string_encoding,
                true,
                None,
            );
            let types = (*instance).component_types();
            let lift = &mut LiftContext::new(cx.0, &options, types, instance);
            let sender_id = TableId::<TransmitSender>::new(u32::load(
                lift,
                InterfaceType::U32,
                &lift.memory()[usize::try_from(params).unwrap()..][..4],
            )?);
            params += 4;
            if !(*instance).handle_table().remove(&(ty, sender_id.rep())) {
                bail!("invalid handle");
            }
            let (sender, entry) = match ty {
                TableIndex::Future(_) => (cx.concurrent_state().table.delete(sender_id)?, None),
                TableIndex::Stream(_) => (
                    *cx.concurrent_state().table.get(sender_id)?,
                    Some((ty, sender_id.rep())),
                ),
            };
            let transmit = cx.concurrent_state().table.get_mut(sender.0)?;
            let new_state = if let ReceiveState::Closed = &transmit.receive {
                ReceiveState::Closed
            } else {
                ReceiveState::Open
            };

            let (status, call) = match mem::replace(&mut transmit.receive, new_state) {
                ReceiveState::GuestReady {
                    ty: receive_ty,
                    flat_abi: receive_flat_abi,
                    options: receive_options,
                    results: receive_results,
                    instance: _,
                    tx: _receive_tx,
                    entry: receive_entry,
                } => {
                    assert_eq!(flat_abi, receive_flat_abi);

                    copy(
                        cx.as_context_mut(),
                        types,
                        instance,
                        flat_abi,
                        ty,
                        &options,
                        params,
                        receive_ty,
                        &receive_options,
                        receive_results,
                    )?;

                    if let Some(entry) = receive_entry {
                        (*instance).handle_table().insert(entry);
                    }

                    (STATUS_DONE, 0)
                }

                ReceiveState::HostReady { accept } => {
                    let lift = &mut LiftContext::new(cx.0, &options, types, instance);
                    accept(Sender::Guest {
                        lift,
                        ty: payload(ty, types),
                        ptr: params,
                    })?;

                    (STATUS_DONE, 0)
                }

                ReceiveState::Open => {
                    assert!(matches!(&transmit.send, SendState::Open));

                    let caller = cx.concurrent_state().guest_task.unwrap();
                    let task = cx
                        .concurrent_state()
                        .table
                        .push_child(HostTask { caller }, caller)?;
                    let (tx, rx) = oneshot::channel();
                    let future = Box::pin(rx.map(move |_| {
                        (
                            task.rep(),
                            Box::new(move |_| Ok(()))
                                as Box<dyn FnOnce(*mut dyn VMStore) -> Result<()>>,
                        )
                    })) as HostTaskFuture;
                    cx.concurrent_state().futures.get_mut().push(future);

                    let transmit = cx.concurrent_state().table.get_mut(sender.0)?;
                    transmit.send = SendState::GuestReady {
                        ty,
                        flat_abi,
                        options,
                        params,
                        results,
                        instance: SendSyncPtr::new(NonNull::new(instance).unwrap()),
                        tx,
                        entry,
                        close: false,
                    };

                    (STATUS_NOT_STARTED, task.rep())
                }

                ReceiveState::Closed => {
                    if let TableIndex::Future(_) = ty {
                        cx.concurrent_state().table.delete(sender.0)?;
                    }

                    cx.concurrent_state().table.delete(sender.0)?;

                    (STATUS_DONE, 0)
                }
            };

            if status == STATUS_DONE {
                let mut lower = LowerContext::new(cx.as_context_mut(), &options, types, instance);
                Ok::<_, Error>(()).store(
                    &mut lower,
                    send_result(ty, types),
                    usize::try_from(results).unwrap(),
                )?;

                if let Some(entry) = entry {
                    (*instance).handle_table().insert(entry);
                }
            }

            Ok((status << 30) | call)
        })
    }
}

fn guest_receive<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TableIndex,
    flat_abi: Option<FlatAbi>,
    params: u32,
    results: u32,
) -> u32 {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                NonNull::new(realloc),
                string_encoding,
                true,
                None,
            );
            let types = (*instance).component_types();
            let lift = &mut LiftContext::new(cx.0, &options, types, instance);
            let receiver_id = TableId::<TransmitReceiver>::new(u32::load(
                lift,
                InterfaceType::U32,
                &lift.memory()[usize::try_from(params).unwrap()..][..4],
            )?);
            if !(*instance).handle_table().remove(&(ty, receiver_id.rep())) {
                bail!("invalid handle");
            }
            let (receiver, entry) = match ty {
                TableIndex::Future(_) => (cx.concurrent_state().table.delete(receiver_id)?, None),
                TableIndex::Stream(_) => (
                    *cx.concurrent_state().table.get(receiver_id)?,
                    Some((ty, receiver_id.rep())),
                ),
            };
            let transmit = cx.concurrent_state().table.get_mut(receiver.0)?;
            let new_state = if let SendState::Closed = &transmit.send {
                SendState::Closed
            } else {
                SendState::Open
            };

            let (status, call) = match mem::replace(&mut transmit.send, new_state) {
                SendState::GuestReady {
                    ty: send_ty,
                    flat_abi: send_flat_abi,
                    options: send_options,
                    params: send_params,
                    instance: _,
                    results: send_results,
                    tx: _send_tx,
                    entry: send_entry,
                    close,
                } => {
                    assert_eq!(flat_abi, send_flat_abi);

                    let mut lower =
                        LowerContext::new(cx.as_context_mut(), &send_options, types, instance);
                    Ok::<_, Error>(()).store(
                        &mut lower,
                        send_result(send_ty, types),
                        usize::try_from(send_results).unwrap(),
                    )?;

                    copy(
                        cx.as_context_mut(),
                        types,
                        instance,
                        flat_abi,
                        send_ty,
                        &send_options,
                        send_params,
                        ty,
                        &options,
                        results,
                    )?;

                    if close {
                        cx.concurrent_state().table.get_mut(receiver.0)?.send = SendState::Closed;
                    } else if let Some(entry) = send_entry {
                        (*instance).handle_table().insert(entry);
                    }

                    (STATUS_DONE, 0)
                }

                SendState::HostReady { accept, close } => {
                    accept(Receiver::Guest {
                        lower: RawLowerContext {
                            store: cx.0.traitobj(),
                            options: &options,
                            types,
                            instance,
                        },
                        ty: receive_result(ty, types),
                        offset: usize::try_from(results).unwrap(),
                    })?;

                    if close {
                        cx.concurrent_state().table.get_mut(receiver.0)?.send = SendState::Closed;
                    }

                    (STATUS_DONE, 0)
                }

                SendState::Open => {
                    assert!(matches!(&transmit.receive, ReceiveState::Open));

                    let caller = cx.concurrent_state().guest_task.unwrap();
                    let task = cx
                        .concurrent_state()
                        .table
                        .push_child(HostTask { caller }, caller)?;
                    let (tx, rx) = oneshot::channel();
                    let future = Box::pin(rx.map(move |_| {
                        (
                            task.rep(),
                            Box::new(move |_| Ok(()))
                                as Box<dyn FnOnce(*mut dyn VMStore) -> Result<()>>,
                        )
                    })) as HostTaskFuture;
                    cx.concurrent_state().futures.get_mut().push(future);

                    let transmit = cx.concurrent_state().table.get_mut(receiver.0)?;
                    transmit.receive = ReceiveState::GuestReady {
                        ty,
                        flat_abi,
                        options,
                        results,
                        instance: SendSyncPtr::new(NonNull::new(instance).unwrap()),
                        tx,
                        entry,
                    };

                    (STATUS_NOT_STARTED, task.rep())
                }

                SendState::Closed => {
                    if let TableIndex::Future(_) = ty {
                        cx.concurrent_state().table.delete(receiver.0)?;
                    }

                    let mut lower = LowerContext::new(cx, &options, types, instance);
                    match ty {
                        TableIndex::Future(ty) => {
                            Err::<(), _>(Error { rep: 0 }).store(
                                &mut lower,
                                InterfaceType::Result(types[ty].receive_result),
                                usize::try_from(results).unwrap(),
                            )?;
                        }
                        TableIndex::Stream(ty) => {
                            Val::Option(None).store(
                                &mut lower,
                                InterfaceType::Option(types[ty].receive_option),
                                usize::try_from(results).unwrap(),
                            )?;
                        }
                    }

                    (STATUS_DONE, 0)
                }
            };

            if status == STATUS_DONE {
                if let Some(entry) = entry {
                    (*instance).handle_table().insert(entry);
                }
            }

            Ok((status << 30) | call)
        })
    }
}

fn guest_drop_sender<T>(vmctx: *mut VMOpaqueContext, ty: TableIndex, sender: u32) {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let cx = StoreContextMut::<T>::from_raw((*instance).store());
            if !(*instance).handle_table().remove(&(ty, sender)) {
                bail!("invalid handle");
            }
            host_close_sender(cx, sender)
        })
    }
}

fn guest_drop_receiver<T>(vmctx: *mut VMOpaqueContext, ty: TableIndex, receiver: u32) {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let cx = StoreContextMut::<T>::from_raw((*instance).store());
            if !(*instance).handle_table().remove(&(ty, receiver)) {
                bail!("invalid handle");
            }
            host_close_receiver(cx, receiver)
        })
    }
}

pub(crate) extern "C" fn future_new<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    ty: TypeFutureTableIndex,
    results: u32,
) {
    guest_new::<T>(vmctx, memory, TableIndex::Future(ty), results)
}

pub(crate) extern "C" fn future_send<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeFutureTableIndex,
    params: u32,
    results: u32,
) -> u32 {
    guest_send::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Future(ty),
        None,
        params,
        results,
    )
}

pub(crate) extern "C" fn future_receive<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeFutureTableIndex,
    params: u32,
    results: u32,
) -> u32 {
    guest_receive::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Future(ty),
        None,
        params,
        results,
    )
}

pub(crate) extern "C" fn future_drop_sender<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    sender: u32,
) {
    guest_drop_sender::<T>(vmctx, TableIndex::Future(ty), sender)
}

pub(crate) extern "C" fn future_drop_receiver<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    receiver: u32,
) {
    guest_drop_receiver::<T>(vmctx, TableIndex::Future(ty), receiver)
}

pub(crate) extern "C" fn stream_new<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    ty: TypeStreamTableIndex,
    results: u32,
) {
    guest_new::<T>(vmctx, memory, TableIndex::Stream(ty), results)
}

pub(crate) extern "C" fn stream_send<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeStreamTableIndex,
    params: u32,
    results: u32,
) -> u32 {
    guest_send::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Stream(ty),
        None,
        params,
        results,
    )
}

pub(crate) extern "C" fn stream_receive<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeStreamTableIndex,
    params: u32,
    results: u32,
) -> u32 {
    guest_receive::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Stream(ty),
        None,
        params,
        results,
    )
}

pub(crate) extern "C" fn stream_drop_sender<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    sender: u32,
) {
    guest_drop_sender::<T>(vmctx, TableIndex::Stream(ty), sender)
}

pub(crate) extern "C" fn stream_drop_receiver<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    receiver: u32,
) {
    guest_drop_receiver::<T>(vmctx, TableIndex::Stream(ty), receiver)
}

pub(crate) extern "C" fn flat_stream_send<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    ty: TypeStreamTableIndex,
    payload_size: u32,
    payload_align: u32,
    params: u32,
    results: u32,
) -> u32 {
    guest_send::<T>(
        vmctx,
        memory,
        realloc,
        StringEncoding::Utf8,
        TableIndex::Stream(ty),
        Some(FlatAbi {
            size: payload_size,
            align: payload_align,
        }),
        params,
        results,
    )
}

pub(crate) extern "C" fn flat_stream_receive<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    ty: TypeStreamTableIndex,
    payload_size: u32,
    payload_align: u32,
    params: u32,
    results: u32,
) -> u32 {
    guest_receive::<T>(
        vmctx,
        memory,
        realloc,
        StringEncoding::Utf8,
        TableIndex::Stream(ty),
        Some(FlatAbi {
            size: payload_size,
            align: payload_align,
        }),
        params,
        results,
    )
}

pub(crate) extern "C" fn error_drop<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeErrorTableIndex,
    error: u32,
) {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let count = if let Some(count) = (*instance).error_table().get_mut(&(ty, error)) {
                assert!(*count > 0);
                *count -= 1;
                *count
            } else {
                bail!("invalid handle");
            };

            if count == 0 {
                (*instance).error_table().remove(&(ty, error));
            }

            Ok(())
        })
    }
}
