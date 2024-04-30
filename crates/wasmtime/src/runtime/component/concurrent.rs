use crate::{
    component::{
        func::{self, Lift as _, LiftContext, Lower as _, LowerContext, Options},
        matching::InstanceType,
        Val,
    },
    AsContextMut, Engine, StoreContextMut, ValRaw,
};
use anyhow::{anyhow, bail, Result};
use futures::{
    channel::oneshot,
    future::{self, Either, FutureExt},
    stream::{FuturesUnordered, ReadyChunks, StreamExt},
};
use once_cell::sync::Lazy;
use std::{
    any::Any,
    cell::UnsafeCell,
    collections::VecDeque,
    future::Future,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    panic::{self, AssertUnwindSafe},
    pin::{pin, Pin},
    ptr::{self, NonNull},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
use table::{Table, TableId};
use wasmtime_environ::component::{
    CanonicalAbiInfo, ComponentTypes, InterfaceType, StringEncoding, TypeErrorTableIndex,
    TypeFuncIndex, TypeFutureIndex, TypeFutureTableIndex, TypeStreamIndex, TypeStreamTableIndex,
    MAX_FLAT_PARAMS,
};
use wasmtime_fiber::{Fiber, Suspend};
use wasmtime_runtime::{
    component::{ComponentInstance, TableIndex, VMComponentContext},
    mpk::{self, ProtectionMask},
    AsyncWasmCallState, PreviousAsyncWasmCallState, SendSyncPtr, Store, VMFuncRef,
    VMMemoryDefinition, VMOpaqueContext,
};

mod table;

/// TODO: add `validate_inbounds` calls where appropriate

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

fn send<
    T: func::Lower + Send + Sync + 'static,
    W: func::Lower + Send + Sync + 'static,
    U: 'static,
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
        .get(TableId::<TransmitSender<U>>::new(rep))?
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
                        Receiver::Guest { lower, ty, offset } => {
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
            options,
            results,
            instance,
            tx: _receive_tx,
            entry,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let ty = match transmit.ty {
                TransmitStateIndex::Future(ty) => InterfaceType::Result(types[ty].receive_result),
                TransmitStateIndex::Stream(ty) => InterfaceType::Option(types[ty].receive_option),
                TransmitStateIndex::Unknown => unreachable!(),
            };
            let lower =
                &mut LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());
            wrap(value).store(lower, ty, usize::try_from(results).unwrap())?;

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

pub fn receive<T: func::Lift + Sync + Send + 'static, U: 'static, S: AsContextMut<Data = U>>(
    mut store: S,
    rep: u32,
) -> Result<oneshot::Receiver<Option<T>>> {
    let mut store = store.as_context_mut();
    let (tx, rx) = oneshot::channel();
    let transmit_id = store
        .concurrent_state()
        .table
        .get(TableId::<TransmitReceiver<U>>::new(rep))?
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
            options,
            params,
            results,
            instance,
            tx: _send_tx,
            entry,
            close,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let ty = match transmit.ty {
                TransmitStateIndex::Future(ty) => types[ty].payload,
                TransmitStateIndex::Stream(ty) => Some(InterfaceType::List(types[ty].list)),
                TransmitStateIndex::Unknown => unreachable!(),
            };
            let send_result = transmit.ty.send_result(types);
            let lift = &mut LiftContext::new(store.0, &options, types, instance.as_ptr());
            _ = tx.send(
                ty.map(|ty| {
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
                send_result,
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

fn close_sender<U: 'static, S: AsContextMut<Data = U>>(mut store: S, rep: u32) -> Result<()> {
    let mut store = store.as_context_mut();
    let sender = store
        .concurrent_state()
        .table
        .delete::<TransmitSender<U>>(TableId::new(rep))?;
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
            options,
            results,
            instance,
            tx: _tx,
            entry,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let ty = transmit.ty;
            let mut lower =
                LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());

            match ty {
                TransmitStateIndex::Future(ty) => {
                    Err::<(), _>(Error { rep: 0 }).store(
                        &mut lower,
                        InterfaceType::Result(types[ty].receive_result),
                        usize::try_from(results).unwrap(),
                    )?;
                }
                TransmitStateIndex::Stream(ty) => {
                    Val::Option(None).store(
                        &mut lower,
                        InterfaceType::Option(types[ty].receive_option),
                        usize::try_from(results).unwrap(),
                    )?;
                }
                TransmitStateIndex::Unknown => unreachable!(),
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

fn close_receiver<U: 'static, S: AsContextMut<Data = U>>(mut store: S, rep: u32) -> Result<()> {
    let mut store = store.as_context_mut();
    let receiver = store
        .concurrent_state()
        .table
        .delete::<TransmitReceiver<U>>(TableId::new(rep))?;
    let transmit = store.concurrent_state().table.get_mut(receiver.0)?;

    transmit.receive = ReceiveState::Closed;

    let new_state = if let SendState::Closed = &transmit.send {
        SendState::Closed
    } else {
        SendState::Open
    };

    match mem::replace(&mut transmit.send, new_state) {
        SendState::GuestReady {
            options,
            params: _,
            instance,
            results,
            tx: _tx,
            entry,
            close,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let ty = transmit.ty;
            let mut lower =
                LowerContext::new(store.as_context_mut(), &options, types, instance.as_ptr());
            Err::<(), _>(Error { rep: 0 }).store(
                &mut lower,
                ty.send_result(types),
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

/// TODO: docs
pub struct FutureSender<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> FutureSender<T> {
    /// TODO: docs
    pub fn send<U: 'static, S: AsContextMut<Data = U>>(self, store: S, value: T) -> Result<()>
    where
        T: func::Lower + Send + Sync + 'static,
    {
        send(store, self.rep, value, |v| Ok::<_, Error>(v))
    }

    pub fn close<U: 'static, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        close_sender(store, self.rep)
    }
}

/// TODO: docs
pub struct FutureReceiver<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> FutureReceiver<T> {
    /// TODO: docs
    pub fn receive<U: 'static, S: AsContextMut<Data = U>>(
        self,
        store: S,
    ) -> Result<oneshot::Receiver<Option<T>>>
    where
        T: func::Lift + Sync + Send + 'static,
    {
        receive(store, self.rep)
    }

    fn lower_to_index<U: 'static>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
    ) -> Result<u32> {
        match ty {
            InterfaceType::Future(dst) => {
                let lower_ty = unsafe { (*cx.instance).component_types()[dst].ty };

                let transmit = cx
                    .store
                    .concurrent_state()
                    .table
                    .get(TableId::<TransmitReceiver<U>>::new(self.rep))?
                    .0;
                let transmit = cx.store.concurrent_state().table.get_mut(transmit)?;
                match transmit.ty {
                    TransmitStateIndex::Future(ty) => {
                        if lower_ty != ty {
                            bail!("mismatched future types");
                        }
                    }
                    TransmitStateIndex::Stream(_) => bail!("expected future, got stream"),
                    TransmitStateIndex::Unknown => {
                        transmit.ty = TransmitStateIndex::Future(lower_ty);
                    }
                }

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
    pub fn close<U: 'static, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        close_receiver(store, self.rep)
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
    fn lower<U: 'static>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
        dst: &mut MaybeUninit<Self::Lower>,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .lower(cx, InterfaceType::U32, dst)
    }

    fn store<U: 'static>(
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
pub fn future<T, U: 'static, S: AsContextMut<Data = U>>(
    mut store: S,
) -> Result<(FutureSender<T>, FutureReceiver<T>)> {
    let mut store = store.as_context_mut();
    let transmit = store.concurrent_state().table.push(TransmitState::<U> {
        ty: TransmitStateIndex::Unknown,
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
    pub fn send<U: 'static, S: AsContextMut<Data = U>>(
        &mut self,
        store: S,
        values: Vec<T>,
    ) -> Result<()>
    where
        T: func::Lower + Send + Sync + 'static,
    {
        send(store, self.rep, values, |v| Some(Ok::<_, Error>(v)))
    }

    pub fn close<U: 'static, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        close_sender(store, self.rep)
    }
}

/// TODO: docs
pub struct StreamReceiver<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> StreamReceiver<T> {
    /// TODO: docs
    pub fn receive<U: 'static, S: AsContextMut<Data = U>>(
        &mut self,
        store: S,
    ) -> Result<oneshot::Receiver<Option<Vec<T>>>>
    where
        T: func::Lift + Sync + Send + 'static,
    {
        receive(store, self.rep)
    }

    fn lower_to_index<U: 'static>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
    ) -> Result<u32> {
        match ty {
            InterfaceType::Stream(dst) => {
                let lower_ty = unsafe { (*cx.instance).component_types()[dst].ty };

                let transmit = cx
                    .store
                    .concurrent_state()
                    .table
                    .get(TableId::<TransmitReceiver<U>>::new(self.rep))?
                    .0;
                let transmit = cx.store.concurrent_state().table.get_mut(transmit)?;
                match transmit.ty {
                    TransmitStateIndex::Stream(ty) => {
                        if lower_ty != ty {
                            bail!("mismatched stream types");
                        }
                    }
                    TransmitStateIndex::Future(_) => bail!("expected stream, got future"),
                    TransmitStateIndex::Unknown => {
                        transmit.ty = TransmitStateIndex::Stream(lower_ty);
                    }
                }

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
    pub fn close<U: 'static, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        close_receiver(store, self.rep)
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
    fn lower<U: 'static>(
        &self,
        cx: &mut LowerContext<'_, U>,
        ty: InterfaceType,
        dst: &mut MaybeUninit<Self::Lower>,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .lower(cx, InterfaceType::U32, dst)
    }

    fn store<U: 'static>(
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
pub fn stream<T, U: 'static, S: AsContextMut<Data = U>>(
    mut store: S,
) -> Result<(StreamSender<T>, StreamReceiver<T>)> {
    let mut store = store.as_context_mut();
    let transmit = store.concurrent_state().table.push(TransmitState::<U> {
        ty: TransmitStateIndex::Unknown,
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
    fn lower<T: 'static>(
        &self,
        cx: &mut LowerContext<'_, T>,
        ty: InterfaceType,
        dst: &mut MaybeUninit<Self::Lower>,
    ) -> Result<()> {
        self.lower_to_index(cx, ty)?
            .lower(cx, InterfaceType::U32, dst)
    }

    fn store<T: 'static>(
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TransmitStateIndex {
    Future(TypeFutureIndex),
    Stream(TypeStreamIndex),
    Unknown,
}

impl TransmitStateIndex {
    fn send_result(self, types: &Arc<ComponentTypes>) -> InterfaceType {
        InterfaceType::Result(match self {
            TransmitStateIndex::Future(ty) => types[ty].send_result,
            TransmitStateIndex::Stream(ty) => types[ty].send_result,
            TransmitStateIndex::Unknown => unreachable!(),
        })
    }
}

struct TransmitState<T> {
    ty: TransmitStateIndex,
    send: SendState<T>,
    receive: ReceiveState,
}

struct TransmitSender<T>(TableId<TransmitState<T>>);

impl<T> Clone for TransmitSender<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T> Copy for TransmitSender<T> {}

struct TransmitReceiver<T>(TableId<TransmitState<T>>);

impl<T> Clone for TransmitReceiver<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T> Copy for TransmitReceiver<T> {}

enum SendState<T> {
    Open,
    GuestReady {
        options: Options,
        params: u32,
        results: u32,
        instance: SendSyncPtr<ComponentInstance>,
        tx: oneshot::Sender<()>,
        entry: Option<(TableIndex, u32)>,
        close: bool,
    },
    HostReady {
        accept: Box<dyn FnOnce(Receiver<'_, T>) -> Result<()> + Send + Sync>,
        close: bool,
    },
    Closed,
}

enum ReceiveState {
    Open,
    GuestReady {
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

enum Receiver<'a, T> {
    Guest {
        lower: &'a mut LowerContext<'a, T>,
        ty: InterfaceType,
        offset: usize,
    },
    Host {
        accept: Box<dyn FnOnce(Box<dyn Any>) -> Result<()>>,
    },
    None,
}

type HostTaskFuture<T> = Pin<
    Box<
        dyn Future<Output = (u32, Box<dyn FnOnce(StoreContextMut<'_, T>) -> Result<()>>)>
            + Send
            + Sync
            + 'static,
    >,
>;

pub struct HostTask<T> {
    caller: TableId<GuestTask<T>>,
}

struct Caller<T> {
    task: TableId<GuestTask<T>>,
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
    guest_task: Option<TableId<GuestTask<T>>>,
    futures: ReadyChunks<FuturesUnordered<HostTaskFuture<T>>>,
    table: Table,
    async_state: AsyncState,
    result: Option<Box<dyn Any + Send + Sync>>,
    sync_task_queue: VecDeque<TableId<GuestTask<T>>>,
}

impl<T> Default for ConcurrentState<T> {
    fn default() -> Self {
        Self {
            guest_task: None,
            table: Table::new(),
            futures: FuturesUnordered::new().ready_chunks(1024),
            async_state: AsyncState {
                current_suspend: UnsafeCell::new(ptr::null()),
                current_poll_cx: UnsafeCell::new(ptr::null_mut()),
            },
            result: None,
            sync_task_queue: VecDeque::new(),
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

pub(crate) fn first_poll<T: 'static, R: Send + 'static>(
    mut store: StoreContextMut<T>,
    future: impl Future<Output = impl FnOnce(StoreContextMut<T>) -> Result<R> + 'static>
        + Send
        + Sync
        + 'static,
    lower: impl FnOnce(StoreContextMut<T>, R) -> Result<()> + Send + Sync + 'static,
) -> Result<Option<TableId<HostTask<T>>>> {
    let caller = store.concurrent_state().guest_task.unwrap();
    let task = store
        .concurrent_state()
        .table
        .push_child(HostTask { caller }, caller)?;
    let mut future = Box::pin(future.map(move |fun| {
        (
            task.rep(),
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
                Box::new(for_any(move |mut store| {
                    let result = fun(store.as_context_mut())?;
                    store.concurrent_state().table.get_mut(caller)?.result =
                        Some(Box::new(result) as _);
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
    guest_task: TableId<GuestTask<T>>,
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
    } else if let Some(fiber) = store
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

fn resume<'a, T: 'static>(
    mut store: StoreContextMut<'a, T>,
    guest_task: TableId<GuestTask<T>>,
    mut fiber: StoreFiber,
) -> Result<StoreContextMut<'a, T>> {
    match unsafe { resume_fiber(&mut fiber, Some(store), Ok(())) } {
        Ok((mut store, result)) => {
            result?;
            store = resume_next_sync_task(store, guest_task)?;
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

fn poll_for_result<'a, T: 'static>(
    mut store: StoreContextMut<'a, T>,
) -> Result<StoreContextMut<'a, T>> {
    let task = store.concurrent_state().guest_task;
    poll_loop(store, move |store| {
        task.map(|task| {
            Ok::<_, anyhow::Error>(store.concurrent_state().table.get(task)?.result.is_none())
        })
        .unwrap_or(Ok(true))
    })
}

fn handle_ready<'a, T: 'static>(
    mut store: StoreContextMut<'a, T>,
    ready: Vec<(u32, Box<dyn FnOnce(StoreContextMut<'_, T>) -> Result<()>>)>,
) -> Result<StoreContextMut<'a, T>> {
    for (task, fun) in ready {
        let task = TableId::<HostTask<T>>::new(task);
        fun(store.as_context_mut())?;
        let caller = store.concurrent_state().table.delete(task)?.caller;
        store = maybe_send_event(store, caller, EVENT_CALL_DONE, task.rep())?;
    }
    Ok(store)
}

fn poll_loop<'a, T: 'static>(
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

fn resume_next_sync_task<'a, T: 'static>(
    mut store: StoreContextMut<'a, T>,
    current_task: TableId<GuestTask<T>>,
) -> Result<StoreContextMut<'a, T>> {
    assert_eq!(
        current_task.rep(),
        store
            .concurrent_state()
            .sync_task_queue
            .pop_front()
            .unwrap()
            .rep()
    );

    if let Some(next) = store.concurrent_state().sync_task_queue.pop_front() {
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
        Ok(store)
    }
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

pub(crate) extern "C" fn async_start<T: 'static>(
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
            lower(cx, storage)?;
            Ok(())
        })
    }
}

pub(crate) extern "C" fn async_return<T: 'static>(
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

            let (result, mut cx) = lift(
                cx,
                mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(storage),
            )?;

            cx.concurrent_state().table.get_mut(guest_task)?.result = result;

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
            let old_task_rep = old_task.map(|v| v.rep());
            let new_task = GuestTask {
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
                        cx = maybe_send_event(
                            cx,
                            TableId::new(rep),
                            EVENT_CALL_STARTED,
                            task.rep(),
                        )?;
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
                        cx = maybe_send_event(
                            cx,
                            TableId::new(rep),
                            EVENT_CALL_RETURNED,
                            task.rep(),
                        )?;
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

                let mut fiber = make_fiber(&mut cx, move |mut cx| {
                    let mut storage = [MaybeUninit::uninit(); MAX_FLAT_PARAMS];
                    let lower = cx
                        .concurrent_state()
                        .table
                        .get_mut(guest_task)?
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
                        .lift_result
                        .take()
                        .unwrap();

                    assert!(cx
                        .concurrent_state()
                        .table
                        .get(guest_task)?
                        .result
                        .is_none());

                    let (result, mut cx) = lift(
                        cx,
                        mem::transmute::<&[MaybeUninit<ValRaw>], &[ValRaw]>(
                            &storage[..result_count],
                        ),
                    )?;
                    cx.concurrent_state().table.get_mut(guest_task)?.result = result;

                    Ok(())
                })?;

                let queue = &mut cx.concurrent_state().sync_task_queue;
                let first_in_queue = queue.is_empty();
                queue.push_back(guest_task);

                if first_in_queue {
                    let mut cx = Some(cx);
                    loop {
                        match resume_fiber(&mut fiber, cx.take(), Ok(())) {
                            Ok((cx, result)) => {
                                result?;
                                break resume_next_sync_task(cx, guest_task)?;
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

fn transmit_new<T: 'static>(
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
            let transmit = cx.concurrent_state().table.push(TransmitState::<T> {
                ty: match ty {
                    TableIndex::Future(ty) => TransmitStateIndex::Future(types[ty].ty),
                    TableIndex::Stream(ty) => TransmitStateIndex::Stream(types[ty].ty),
                },
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

unsafe fn copy<T: 'static>(
    mut cx: StoreContextMut<'_, T>,
    ty: TransmitStateIndex,
    types: &Arc<ComponentTypes>,
    instance: *mut ComponentInstance,
    send_options: &Options,
    send_params: u32,
    receive_options: &Options,
    receive_results: u32,
) -> Result<()> {
    match ty {
        TransmitStateIndex::Future(ty) => {
            let ty = &types[ty];
            let val = ty
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
                InterfaceType::Result(ty.receive_result),
                usize::try_from(receive_results).unwrap(),
            )?;
        }
        TransmitStateIndex::Stream(ty) => {
            let ty = &types[ty];
            let lift = &mut LiftContext::new(cx.0, send_options, types, instance);
            let val = Val::load(
                lift,
                InterfaceType::List(ty.list),
                &lift.memory()[usize::try_from(send_params).unwrap()..][..usize::try_from(
                    types.canonical_abi(&InterfaceType::List(ty.list)).size32,
                )
                .unwrap()],
            )?;

            let mut lower =
                LowerContext::new(cx.as_context_mut(), receive_options, types, instance);
            Val::Option(Some(Box::new(Val::Result(Ok(Some(Box::new(val))))))).store(
                &mut lower,
                InterfaceType::Option(ty.receive_option),
                usize::try_from(receive_results).unwrap(),
            )?;
        }
        TransmitStateIndex::Unknown => unreachable!(),
    }

    Ok(())
}

fn transmit_send<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    table: TableIndex,
    mut params: u32,
    results: u32,
    call: u32,
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
            let sender_id = TableId::<TransmitSender<T>>::new(u32::load(
                lift,
                InterfaceType::U32,
                &lift.memory()[usize::try_from(params).unwrap()..][..4],
            )?);
            params += 4;
            if !(*instance).handle_table().remove(&(table, sender_id.rep())) {
                bail!("invalid handle");
            }
            let (sender, ty, entry) = match table {
                TableIndex::Future(ty) => (
                    cx.concurrent_state().table.delete(sender_id)?,
                    TransmitStateIndex::Future(types[ty].ty),
                    None,
                ),
                TableIndex::Stream(ty) => (
                    *cx.concurrent_state().table.get(sender_id)?,
                    TransmitStateIndex::Stream(types[ty].ty),
                    Some((table, sender_id.rep())),
                ),
            };
            let transmit = cx.concurrent_state().table.get_mut(sender.0)?;
            if ty != transmit.ty {
                bail!("transmit type mismatch");
            }
            let on_done = || {
                if let Some(entry) = entry {
                    (*instance).handle_table().insert(entry);
                }
            };
            let new_state = if let ReceiveState::Closed = &transmit.receive {
                ReceiveState::Closed
            } else {
                ReceiveState::Open
            };

            match mem::replace(&mut transmit.receive, new_state) {
                ReceiveState::GuestReady {
                    options: receive_options,
                    results: receive_results,
                    instance: _,
                    tx: _receive_tx,
                    entry: receive_entry,
                } => {
                    let mut lower =
                        LowerContext::new(cx.as_context_mut(), &options, types, instance);
                    Ok::<_, Error>(()).store(
                        &mut lower,
                        ty.send_result(types),
                        usize::try_from(results).unwrap(),
                    )?;

                    copy(
                        cx.as_context_mut(),
                        ty,
                        types,
                        instance,
                        &options,
                        params,
                        &receive_options,
                        receive_results,
                    )?;

                    if let Some(entry) = receive_entry {
                        (*instance).handle_table().insert(entry);
                    }

                    on_done();

                    Ok(STATUS_DONE)
                }

                ReceiveState::HostReady { accept } => {
                    let lift = &mut LiftContext::new(cx.0, &options, types, instance);
                    accept(Sender::Guest {
                        lift,
                        ty: match ty {
                            TransmitStateIndex::Future(ty) => types[ty].payload,
                            TransmitStateIndex::Stream(ty) => {
                                Some(InterfaceType::List(types[ty].list))
                            }
                            TransmitStateIndex::Unknown => unreachable!(),
                        },
                        ptr: params,
                    })?;

                    on_done();

                    Ok(STATUS_DONE)
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
                            Box::new(for_any(move |_| Ok(())))
                                as Box<dyn FnOnce(StoreContextMut<T>) -> Result<()>>,
                        )
                    })) as HostTaskFuture<T>;
                    cx.concurrent_state().futures.get_mut().push(future);

                    let transmit = cx.concurrent_state().table.get_mut(sender.0)?;
                    transmit.send = SendState::GuestReady {
                        options,
                        params,
                        results,
                        instance: SendSyncPtr::new(NonNull::new(instance).unwrap()),
                        tx,
                        entry,
                        close: false,
                    };

                    let ptr = func::validate_inbounds::<u32>(
                        options.memory_mut(cx.0),
                        &ValRaw::u32(call),
                    )?;
                    let mut lower = LowerContext::new(cx, &options, types, instance);
                    task.rep().store(&mut lower, InterfaceType::U32, ptr)?;

                    Ok(STATUS_NOT_STARTED)
                }

                ReceiveState::Closed => {
                    if let TransmitStateIndex::Future(_) = ty {
                        cx.concurrent_state().table.delete(sender.0)?;
                    }

                    cx.concurrent_state().table.delete(sender.0)?;

                    let mut lower = LowerContext::new(cx, &options, types, instance);
                    Err::<(), _>(Error { rep: 0 }).store(
                        &mut lower,
                        ty.send_result(types),
                        usize::try_from(results).unwrap(),
                    )?;

                    on_done();

                    Ok(STATUS_DONE)
                }
            }
        })
    }
}

fn transmit_receive<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    table: TableIndex,
    params: u32,
    results: u32,
    call: u32,
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
            let receiver_id = TableId::<TransmitReceiver<T>>::new(u32::load(
                lift,
                InterfaceType::U32,
                &lift.memory()[usize::try_from(params).unwrap()..][..4],
            )?);
            if !(*instance)
                .handle_table()
                .remove(&(table, receiver_id.rep()))
            {
                bail!("invalid handle");
            }
            let (receiver, ty, entry) = match table {
                TableIndex::Future(ty) => (
                    cx.concurrent_state().table.delete(receiver_id)?,
                    TransmitStateIndex::Future(types[ty].ty),
                    None,
                ),
                TableIndex::Stream(ty) => (
                    *cx.concurrent_state().table.get(receiver_id)?,
                    TransmitStateIndex::Stream(types[ty].ty),
                    Some((table, receiver_id.rep())),
                ),
            };
            let transmit = cx.concurrent_state().table.get_mut(receiver.0)?;
            if ty != transmit.ty {
                bail!("transmit type mismatch");
            }
            let on_done = || {
                if let Some(entry) = entry {
                    (*instance).handle_table().insert(entry);
                }
            };
            let new_state = if let SendState::Closed = &transmit.send {
                SendState::Closed
            } else {
                SendState::Open
            };

            match mem::replace(&mut transmit.send, new_state) {
                SendState::GuestReady {
                    options: send_options,
                    params: send_params,
                    instance: _,
                    results: send_results,
                    tx: _send_tx,
                    entry: send_entry,
                    close,
                } => {
                    let mut lower =
                        LowerContext::new(cx.as_context_mut(), &send_options, types, instance);
                    Ok::<_, Error>(()).store(
                        &mut lower,
                        ty.send_result(types),
                        usize::try_from(send_results).unwrap(),
                    )?;

                    copy(
                        cx.as_context_mut(),
                        ty,
                        types,
                        instance,
                        &send_options,
                        send_params,
                        &options,
                        results,
                    )?;

                    if close {
                        cx.concurrent_state().table.get_mut(receiver.0)?.send = SendState::Closed;
                    } else if let Some(entry) = send_entry {
                        (*instance).handle_table().insert(entry);
                    }

                    on_done();

                    Ok(STATUS_DONE)
                }

                SendState::HostReady { accept, close } => {
                    let mut lower =
                        LowerContext::new(cx.as_context_mut(), &options, types, instance);
                    accept(Receiver::Guest {
                        lower: &mut lower,
                        ty: match ty {
                            TransmitStateIndex::Future(ty) => {
                                InterfaceType::Result(types[ty].receive_result)
                            }
                            TransmitStateIndex::Stream(ty) => {
                                InterfaceType::Option(types[ty].receive_option)
                            }
                            TransmitStateIndex::Unknown => unreachable!(),
                        },
                        offset: usize::try_from(results).unwrap(),
                    })?;

                    if close {
                        cx.concurrent_state().table.get_mut(receiver.0)?.send = SendState::Closed;
                    }

                    on_done();

                    Ok(STATUS_DONE)
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
                            Box::new(for_any(move |_| Ok(())))
                                as Box<dyn FnOnce(StoreContextMut<T>) -> Result<()>>,
                        )
                    })) as HostTaskFuture<T>;
                    cx.concurrent_state().futures.get_mut().push(future);

                    let transmit = cx.concurrent_state().table.get_mut(receiver.0)?;
                    transmit.receive = ReceiveState::GuestReady {
                        options,
                        results,
                        instance: SendSyncPtr::new(NonNull::new(instance).unwrap()),
                        tx,
                        entry,
                    };

                    let ptr = func::validate_inbounds::<u32>(
                        options.memory_mut(cx.0),
                        &ValRaw::u32(call),
                    )?;
                    let mut lower = LowerContext::new(cx, &options, types, instance);
                    task.rep().store(&mut lower, InterfaceType::U32, ptr)?;

                    Ok(STATUS_NOT_STARTED)
                }

                SendState::Closed => {
                    if let TransmitStateIndex::Future(_) = ty {
                        cx.concurrent_state().table.delete(receiver.0)?;
                    }

                    let mut lower = LowerContext::new(cx, &options, types, instance);
                    match ty {
                        TransmitStateIndex::Future(ty) => {
                            Err::<(), _>(Error { rep: 0 }).store(
                                &mut lower,
                                InterfaceType::Result(types[ty].receive_result),
                                usize::try_from(results).unwrap(),
                            )?;
                        }
                        TransmitStateIndex::Stream(ty) => {
                            Val::Option(None).store(
                                &mut lower,
                                InterfaceType::Option(types[ty].receive_option),
                                usize::try_from(results).unwrap(),
                            )?;
                        }
                        TransmitStateIndex::Unknown => unreachable!(),
                    }

                    on_done();

                    Ok(STATUS_DONE)
                }
            }
        })
    }
}

fn transmit_drop_sender<T: 'static>(vmctx: *mut VMOpaqueContext, ty: TableIndex, sender_rep: u32) {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let types = (*instance).component_types();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            if !(*instance).handle_table().remove(&(ty, sender_rep)) {
                bail!("invalid handle");
            }
            let sender = *cx
                .concurrent_state()
                .table
                .get::<TransmitSender<T>>(TableId::new(sender_rep))?;
            let ty = match ty {
                TableIndex::Future(ty) => TransmitStateIndex::Future(types[ty].ty),
                TableIndex::Stream(ty) => TransmitStateIndex::Stream(types[ty].ty),
            };
            if ty != cx.concurrent_state().table.get(sender.0)?.ty {
                bail!("transmit type mismatch");
            }
            close_sender(cx, sender_rep)
        })
    }
}

fn transmit_drop_receiver<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    ty: TableIndex,
    receiver_rep: u32,
) {
    unsafe {
        handle_result(|| {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let types = (*instance).component_types();
            let mut cx = StoreContextMut::<T>::from_raw((*instance).store());
            if !(*instance).handle_table().remove(&(ty, receiver_rep)) {
                bail!("invalid handle");
            }
            let receiver = *cx
                .concurrent_state()
                .table
                .get::<TransmitReceiver<T>>(TableId::new(receiver_rep))?;
            let ty = match ty {
                TableIndex::Future(ty) => TransmitStateIndex::Future(types[ty].ty),
                TableIndex::Stream(ty) => TransmitStateIndex::Stream(types[ty].ty),
            };
            if ty != cx.concurrent_state().table.get(receiver.0)?.ty {
                bail!("transmit type mismatch");
            }
            close_receiver(cx, receiver_rep)
        })
    }
}

pub(crate) extern "C" fn future_new<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    ty: TypeFutureTableIndex,
    results: u32,
) {
    transmit_new::<T>(vmctx, memory, TableIndex::Future(ty), results)
}

pub(crate) extern "C" fn future_send<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeFutureTableIndex,
    params: u32,
    results: u32,
    call: u32,
) -> u32 {
    transmit_send::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Future(ty),
        params,
        results,
        call,
    )
}

pub(crate) extern "C" fn future_receive<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeFutureTableIndex,
    params: u32,
    results: u32,
    call: u32,
) -> u32 {
    transmit_receive::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Future(ty),
        params,
        results,
        call,
    )
}

pub(crate) extern "C" fn future_drop_sender<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    sender: u32,
) {
    transmit_drop_sender::<T>(vmctx, TableIndex::Future(ty), sender)
}

pub(crate) extern "C" fn future_drop_receiver<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    receiver: u32,
) {
    transmit_drop_receiver::<T>(vmctx, TableIndex::Future(ty), receiver)
}

pub(crate) extern "C" fn stream_new<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    ty: TypeStreamTableIndex,
    results: u32,
) {
    transmit_new::<T>(vmctx, memory, TableIndex::Stream(ty), results)
}

pub(crate) extern "C" fn stream_send<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeStreamTableIndex,
    params: u32,
    results: u32,
    call: u32,
) -> u32 {
    transmit_send::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Stream(ty),
        params,
        results,
        call,
    )
}

pub(crate) extern "C" fn stream_receive<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    ty: TypeStreamTableIndex,
    params: u32,
    results: u32,
    call: u32,
) -> u32 {
    transmit_receive::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Stream(ty),
        params,
        results,
        call,
    )
}

pub(crate) extern "C" fn stream_drop_sender<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    sender: u32,
) {
    transmit_drop_sender::<T>(vmctx, TableIndex::Stream(ty), sender)
}

pub(crate) extern "C" fn stream_drop_receiver<T: 'static>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    receiver: u32,
) {
    transmit_drop_receiver::<T>(vmctx, TableIndex::Stream(ty), receiver)
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

pub(crate) fn enter<T: Send + 'static>(
    mut store: StoreContextMut<T>,
    lower_params: Lower<T>,
    lift_result: Lift<T>,
) -> Result<()> {
    assert!(store.concurrent_state().guest_task.is_none());

    let guest_task = store.concurrent_state().table.push(GuestTask {
        lower_params: Some(lower_params),
        lift_result: Some(lift_result),
        result: None,
        callback: None,
        caller: None,
        fiber: None,
    })?;
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
        store.concurrent_state().table.get_mut(guest_task)?.callback =
            Some((SendSyncPtr::new(callback), guest_context));

        store = poll_for_result(store)?;
    }

    let guest_task = store.concurrent_state().guest_task.unwrap();
    if let Some(result) = store
        .concurrent_state()
        .table
        .get_mut(guest_task)?
        .result
        .take()
    {
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

pub(crate) async fn poll<'a, T: Send + 'static>(
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
        if let Some(ready) = store.concurrent_state().futures.next().await {
            store = handle_ready(store, ready)?;
        } else {
            break;
        }
    }

    Ok(store)
}

pub(crate) async fn poll_until<'a, T: Send + 'static, U>(
    mut store: StoreContextMut<'a, T>,
    future: impl Future<Output = U>,
) -> Result<(StoreContextMut<'a, T>, U)> {
    let mut future = Box::pin(future);
    loop {
        let ready = pin!(store.concurrent_state().futures.next());

        match future::select(ready, future).await {
            Either::Left((None, future)) => break Ok((store, future.await)),
            Either::Left((Some(ready), future_again)) => {
                store = handle_ready(store, ready)?;
                future = future_again;
            }
            Either::Right((result, _)) => break Ok((store, result)),
        }
    }
}
