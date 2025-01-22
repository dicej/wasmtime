use {
    super::{
        call_host_and_handle_result, table::TableId, Event, GuestTask, HostTaskFuture,
        HostTaskResult, Promise,
    },
    crate::{
        component::{
            func::{self, Lift, LiftContext, LowerContext, Options},
            matching::InstanceType,
            values::{ErrorContextAny, FutureAny, StreamAny},
            Lower, Val, WasmList, WasmStr,
        },
        vm::{
            component::{
                ComponentInstance, ErrorContextState, GlobalErrorContextRefCount,
                LocalErrorContextRefCount, StateTable, StreamFutureState, VMComponentContext,
                WaitableState,
            },
            SendSyncPtr, VMFuncRef, VMMemoryDefinition, VMOpaqueContext, VMStore,
        },
        AsContextMut, StoreContextMut, ValRaw,
    },
    anyhow::{anyhow, bail, ensure, Context, Result},
    futures::{
        channel::oneshot,
        future::{self, FutureExt},
    },
    std::{
        any::Any,
        boxed::Box,
        marker::PhantomData,
        mem::{self, MaybeUninit},
        ptr::NonNull,
        string::ToString,
        sync::Arc,
        vec::Vec,
    },
    wasmtime_environ::component::{
        CanonicalAbiInfo, ComponentTypes, InterfaceType, StringEncoding,
        TypeComponentGlobalErrorContextTableIndex, TypeComponentLocalErrorContextTableIndex,
        TypeFutureTableIndex, TypeStreamTableIndex,
    },
};

const BLOCKED: usize = 0xffff_ffff;
const CLOSED: usize = 0x8000_0000;

#[derive(Copy, Clone, Debug)]
enum TableIndex {
    Stream(TypeStreamTableIndex),
    Future(TypeFutureTableIndex),
}

fn payload(ty: TableIndex, types: &Arc<ComponentTypes>) -> Option<InterfaceType> {
    match ty {
        TableIndex::Future(ty) => types[types[ty].ty].payload,
        TableIndex::Stream(ty) => types[types[ty].ty].payload,
    }
}

fn state_table(instance: &mut ComponentInstance, ty: TableIndex) -> &mut StateTable<WaitableState> {
    let runtime_instance = match ty {
        TableIndex::Stream(ty) => instance.component_types()[ty].instance,
        TableIndex::Future(ty) => instance.component_types()[ty].instance,
    };
    &mut instance.component_waitable_tables()[runtime_instance]
}

fn push_event<T>(
    mut store: StoreContextMut<T>,
    rep: u32,
    event: Event,
    param: usize,
    caller: TableId<GuestTask>,
) {
    store
        .concurrent_state()
        .futures
        .get_mut()
        .push(Box::pin(future::ready((
            rep,
            Box::new(move |_| {
                Ok(HostTaskResult {
                    event,
                    param: u32::try_from(param).unwrap(),
                    caller,
                })
            })
                as Box<dyn FnOnce(*mut dyn VMStore) -> Result<HostTaskResult> + Send + Sync>,
        ))) as HostTaskFuture);
}

fn get_mut_by_index(
    instance: &mut ComponentInstance,
    ty: TableIndex,
    index: u32,
) -> Result<(u32, &mut StreamFutureState)> {
    get_mut_by_index_from(state_table(instance, ty), ty, index)
}

fn get_mut_by_index_from(
    state_table: &mut StateTable<WaitableState>,
    ty: TableIndex,
    index: u32,
) -> Result<(u32, &mut StreamFutureState)> {
    Ok(match ty {
        TableIndex::Stream(ty) => {
            let (rep, WaitableState::Stream(actual_ty, state)) =
                state_table.get_mut_by_index(index)?
            else {
                bail!("invalid stream handle");
            };
            if *actual_ty != ty {
                bail!("invalid stream handle");
            }
            (rep, state)
        }
        TableIndex::Future(ty) => {
            let (rep, WaitableState::Future(actual_ty, state)) =
                state_table.get_mut_by_index(index)?
            else {
                bail!("invalid future handle");
            };
            if *actual_ty != ty {
                bail!("invalid future handle");
            }
            (rep, state)
        }
    })
}

fn waitable_state(ty: TableIndex, state: StreamFutureState) -> WaitableState {
    match ty {
        TableIndex::Stream(ty) => WaitableState::Stream(ty, state),
        TableIndex::Future(ty) => WaitableState::Future(ty, state),
    }
}

fn accept<T: func::Lower + Send + Sync + 'static, U>(
    values: Vec<T>,
    mut offset: usize,
    transmit_id: TableId<TransmitState>,
    tx: oneshot::Sender<()>,
) -> impl FnOnce(Reader) -> Result<usize> + Send + Sync + 'static {
    move |reader| {
        let count = match reader {
            Reader::Guest {
                lower:
                    RawLowerContext {
                        store,
                        options,
                        types,
                        instance,
                    },
                ty,
                address,
                count,
            } => {
                let mut store = unsafe { StoreContextMut::<U>(&mut *store.cast()) };
                let lower = &mut unsafe {
                    LowerContext::new(store.as_context_mut(), options, types, instance)
                };
                if address % usize::try_from(T::ALIGN32)? != 0 {
                    bail!("read pointer not aligned");
                }
                lower
                    .as_slice_mut()
                    .get_mut(address..)
                    .and_then(|b| b.get_mut(..T::SIZE32 * count))
                    .ok_or_else(|| anyhow::anyhow!("read pointer out of bounds of memory"))?;

                let count = values.len().min(usize::try_from(count).unwrap());

                if let Some(ty) = payload(ty, types) {
                    T::store_list(lower, ty, address, &values[offset..][..count])?;
                }
                offset += count;

                if offset < values.len() {
                    let transmit = store.concurrent_state().table.get_mut(transmit_id)?;
                    assert!(matches!(&transmit.write, WriteState::Open));

                    transmit.write = WriteState::HostReady {
                        accept: Box::new(accept::<T, U>(values, offset, transmit_id, tx)),
                        close: false,
                        err_ctx: 0,
                    };
                }

                count
            }
            Reader::Host { accept } => {
                assert!(offset == 0); // todo: do we need to handle offset != 0?
                let count = values.len();
                accept(Box::new(values))?;

                count
            }
            Reader::None => 0,
        };

        Ok(count)
    }
}

/// Write a waitable value from the host
///
/// # Arguments
///
/// * `store` - the engine store
/// * `transmit_rep` - Global representation of the transmit object that will be modified
/// * `values` - List of values that should be written
/// * `close` - Whether the transmit should be closed after write
/// * `err_ctx` - The error context handle (`0` if no error was written)
///
fn host_write<T: func::Lower + Send + Sync + 'static, U, S: AsContextMut<Data = U>>(
    mut store: S,
    transmit_rep: u32,
    values: Vec<T>,
    mut close: bool,
    err_ctx: u32,
) -> Result<oneshot::Receiver<()>> {
    let mut store = store.as_context_mut();
    let (tx, rx) = oneshot::channel();
    let transmit_id = TableId::<TransmitState>::new(transmit_rep);
    let mut offset = 0;

    loop {
        let transmit = store
            .concurrent_state()
            .table
            .get_mut(transmit_id)
            .with_context(|| format!("retrieving state for transmit [{transmit_rep}]"))?;

        let new_state = if let ReadState::Closed(err_ctx) = &transmit.read {
            ReadState::Closed(*err_ctx)
        } else {
            ReadState::Open
        };

        match mem::replace(&mut transmit.read, new_state) {
            ReadState::Open => {
                assert!(matches!(&transmit.write, WriteState::Open));

                transmit.write = WriteState::HostReady {
                    accept: Box::new(accept::<T, U>(values, offset, transmit_id, tx)),
                    close,
                    err_ctx,
                };
                close = false;
            }

            ReadState::GuestReady {
                ty,
                flat_abi: _,
                options,
                address,
                count,
                instance,
                handle,
                caller,
                ..
            } => unsafe {
                let types = (*instance.as_ptr()).component_types();
                let lower = &mut LowerContext::new(
                    store.as_context_mut(),
                    &options,
                    types,
                    instance.as_ptr(),
                );
                if address % usize::try_from(T::ALIGN32)? != 0 {
                    bail!("read pointer not aligned");
                }
                lower
                    .as_slice_mut()
                    .get_mut(address..)
                    .and_then(|b| b.get_mut(..T::SIZE32 * count))
                    .ok_or_else(|| anyhow::anyhow!("read pointer out of bounds of memory"))?;

                let count = values.len().min(count);
                if let Some(ty) = payload(ty, types) {
                    T::store_list(lower, ty, address, &values[offset..][..count])?;
                }
                offset += count;

                log::trace!("remove read child of {}: {transmit_rep}", caller.rep());
                store
                    .concurrent_state()
                    .table
                    .remove_child(transmit_id, caller)?;

                *get_mut_by_index(&mut *instance.as_ptr(), ty, handle)?.1 = StreamFutureState::Read;

                push_event(
                    store.as_context_mut(),
                    transmit_rep,
                    match ty {
                        TableIndex::Future(_) => Event::FutureRead,
                        TableIndex::Stream(_) => Event::StreamRead,
                    },
                    count,
                    caller,
                );

                if offset < values.len() {
                    continue;
                }
            },

            ReadState::HostReady { accept } => {
                accept(Writer::Host {
                    values: Box::new(values),
                    err_ctx, // TODO: redundant, err_ctx is used after the close later
                })?;
            }

            ReadState::Closed(_) => {}
        }

        if close {
            host_close_writer(store, transmit_rep, err_ctx)?;
        }

        break Ok(rx);
    }
}

pub fn host_read<T: func::Lift + Sync + Send + 'static, U, S: AsContextMut<Data = U>>(
    mut store: S,
    rep: u32,
) -> Result<oneshot::Receiver<Option<Vec<T>>>> {
    let mut store = store.as_context_mut();
    let (tx, rx) = oneshot::channel();
    let transmit_id = TableId::<TransmitState>::new(rep);
    let transmit = store
        .concurrent_state()
        .table
        .get_mut(transmit_id)
        .with_context(|| rep.to_string())?;

    let new_state = if let WriteState::Closed(maybe_err_ctx) = &transmit.write {
        WriteState::Closed(*maybe_err_ctx)
    } else {
        WriteState::Open
    };

    match mem::replace(&mut transmit.write, new_state) {
        WriteState::Open => {
            assert!(matches!(&transmit.read, ReadState::Open));

            transmit.read = ReadState::HostReady {
                accept: Box::new(move |writer| {
                    Ok(match writer {
                        Writer::Guest {
                            lift,
                            ty,
                            address,
                            count,
                            ..
                        } => {
                            _ = tx.send(
                                ty.map(|ty| {
                                    if address % usize::try_from(T::ALIGN32)? != 0 {
                                        bail!("write pointer not aligned");
                                    }
                                    lift.memory()
                                        .get(address..)
                                        .and_then(|b| b.get(..T::SIZE32 * count))
                                        .ok_or_else(|| {
                                            anyhow::anyhow!("write pointer out of bounds of memory")
                                        })?;

                                    let list = &WasmList::new(address, count, lift, ty)?;
                                    T::load_list(lift, list)
                                })
                                .transpose()?,
                            );
                            count
                        }
                        Writer::Host { values, err_ctx } => {
                            let values = *values
                                .downcast::<Vec<T>>()
                                .map_err(|_| anyhow!("transmit type mismatch"))?;
                            let count = values.len();
                            _ = tx.send(Some(values));
                            count
                        }
                        Writer::None => 0,
                    })
                }),
            };
        }

        WriteState::GuestReady {
            ty,
            flat_abi: _,
            options,
            address,
            count,
            instance,
            handle,
            caller,
            close,
            err_ctx,
        } => unsafe {
            let types = (*instance.as_ptr()).component_types();
            let lift = &mut LiftContext::new(store.0, &options, types, instance.as_ptr());
            _ = tx.send(
                payload(ty, types)
                    .map(|ty| {
                        let list = &WasmList::new(address, count, lift, ty)?;
                        T::load_list(lift, list)
                    })
                    .transpose()?,
            );

            log::trace!(
                "remove write child of {}: {}",
                caller.rep(),
                transmit_id.rep()
            );
            store
                .concurrent_state()
                .table
                .remove_child(transmit_id, caller)?;

            if close {
                store.concurrent_state().table.get_mut(transmit_id)?.write =
                    WriteState::Closed(err_ctx);
            } else {
                *get_mut_by_index(&mut *instance.as_ptr(), ty, handle)?.1 =
                    StreamFutureState::Write;
            }

            push_event(
                store,
                transmit_id.rep(),
                match ty {
                    TableIndex::Future(_) => Event::FutureWrite,
                    TableIndex::Stream(_) => Event::StreamWrite,
                },
                count,
                caller,
            );
        },

        WriteState::HostReady {
            accept,
            close,
            err_ctx,
        } => {
            accept(Reader::Host {
                accept: Box::new(move |any| {
                    _ = tx.send(Some(
                        *any.downcast()
                            .map_err(|_| anyhow!("transmit type mismatch"))?,
                    ));
                    Ok(())
                }),
            })?;

            if close {
                store.concurrent_state().table.get_mut(transmit_id)?.write =
                    WriteState::Closed(err_ctx);
            }
        }

        WriteState::Closed(_) => {
            host_close_reader(store, rep)?;
        }
    }

    Ok(rx)
}

fn host_cancel_write<U, S: AsContextMut<Data = U>>(mut store: S, rep: u32) -> Result<u32> {
    let mut store = store.as_context_mut();
    let transmit_id = TableId::<TransmitState>::new(rep);
    let transmit = store.concurrent_state().table.get_mut(transmit_id)?;

    match &transmit.write {
        WriteState::GuestReady { caller, .. } => {
            let caller = *caller;
            transmit.write = WriteState::Open;
            store
                .concurrent_state()
                .table
                .remove_child(transmit_id, caller)?;
        }

        WriteState::HostReady { .. } => {
            transmit.write = WriteState::Open;
        }

        WriteState::Open | WriteState::Closed(_) => {
            bail!("stream or future write canceled when no write is pending")
        }
    }

    log::trace!("canceled write {rep}");

    Ok(0)
}

fn host_cancel_read<U, S: AsContextMut<Data = U>>(mut store: S, rep: u32) -> Result<u32> {
    let mut store = store.as_context_mut();
    let transmit_id = TableId::<TransmitState>::new(rep);
    let transmit = store.concurrent_state().table.get_mut(transmit_id)?;

    match &transmit.read {
        ReadState::GuestReady { caller, .. } => {
            let caller = *caller;
            transmit.read = ReadState::Open;
            store
                .concurrent_state()
                .table
                .remove_child(transmit_id, caller)?;
        }

        ReadState::HostReady { .. } => {
            transmit.read = ReadState::Open;
        }

        ReadState::Open | ReadState::Closed(_) => {
            bail!("stream or future read canceled when no read is pending")
        }
    }

    log::trace!("canceled read {rep}");

    Ok(0)
}

/// Close the writer end of a Future or Stream
///
/// # Arguments
///
/// * `store` - the store for the component
/// * `transmit_rep` - A global-component-level representation of the transmit state for the writer that should be closed
/// * `err_ctx` - An optional error context to pass along as the final value of the writer (`0` if none)
///
fn host_close_writer<U, S: AsContextMut<Data = U>>(
    mut store: S,
    transmit_rep: u32,
    err_ctx: u32,
) -> Result<()> {
    let mut store = store.as_context_mut();
    let transmit_id = TableId::<TransmitState>::new(transmit_rep);
    let transmit = store.concurrent_state().table.get_mut(transmit_id)?;

    // Update transmit write state for the writer
    match &mut transmit.write {
        // For guest-level streams that were waiting for a write, we must update to close on the *next* read.
        WriteState::GuestReady {
            close,
            err_ctx: err_ctx_ref,
            ..
        } => {
            *close = true;
            *err_ctx_ref = err_ctx;
        }

        // For host-level streams that were waiting for a write, we must update to close on the *next* read.
        WriteState::HostReady {
            close,
            err_ctx: err_ctx_ref,
            ..
        } => {
            *close = true;
            *err_ctx_ref = err_ctx;
        }

        // If the write state was simply opened (and a read has not been attempted), we can immediately close
        v @ WriteState::Open => {
            *v = WriteState::Closed(err_ctx);
        }

        // It should be impossible to double-close a writable
        WriteState::Closed(_) => unreachable!("write state is already closed"),
    }

    // If the existing read state is closed, then there's nothing to read
    // and we can keep it that way.
    //
    // If the read state was any other state, then we must set the new state to open
    // to indicate that there *is* data to be read
    let new_state = if let ReadState::Closed(read_err_ctx) = &transmit.read {
        ReadState::Closed(*read_err_ctx)
    } else {
        ReadState::Open
    };

    // Swap in the new read state
    match mem::replace(&mut transmit.read, new_state) {
        // If the guest was ready to read, then we cannot close the reader (or writer)
        // we must deliver the event, and update the state associated with the handle to
        // represent that a read must be performed
        ReadState::GuestReady {
            ty,
            instance,
            handle,
            caller,
            ..
        } => unsafe {
            // Ensure the final read of the guest is queued, with appropriate closure indicator
            push_event(
                store,
                transmit_id.rep(),
                match ty {
                    TableIndex::Future(_) => Event::FutureRead,
                    TableIndex::Stream(_) => Event::StreamRead,
                },
                CLOSED | err_ctx as usize,
                caller,
            );

            *get_mut_by_index(&mut *instance.as_ptr(), ty, handle)?.1 = StreamFutureState::Read;
        },

        // If the the host was ready to read, and the writer end is being closed (host->host write?)
        // we can accept but discard the write, and close the reader immediately
        ReadState::HostReady { accept } => {
            accept(Writer::None)?;
            host_close_reader(store, transmit_rep)?;
        }

        // If the read state is open, then there are no registered readers of the stream/future
        //
        // we can delay delivering a final value
        ReadState::Open => {}

        // If the read state was already closed, then we can remove the transmit state completely
        // (both writer and reader have been closed)
        ReadState::Closed(_) => {
            log::trace!("host_close_writer delete {transmit_rep}");
            store.concurrent_state().table.delete(transmit_id)?;
        }
    }
    Ok(())
}

/// Close the reader end of a Future or Stream
///
/// # Arguments
///
/// * `store` - the store for the component
/// * `transmit_rep` - A global-component-level representation of the transmit state for the reader that should be closed
/// * `err_ctx` - An optional error context to pass along as the final value of the reader (`0` if none)
///
fn host_close_reader<U, S: AsContextMut<Data = U>>(mut store: S, transmit_rep: u32) -> Result<()> {
    let mut store = store.as_context_mut();
    let transmit_id = TableId::<TransmitState>::new(transmit_rep);
    let transmit = store.concurrent_state().table.get_mut(transmit_id)?;

    transmit.read = ReadState::Closed(0);

    // If the write end is already closed, it should stay closed,
    // otherwise, it should be opened.
    let new_state = if let WriteState::Closed(err_ctx) = &transmit.write {
        WriteState::Closed(*err_ctx)
    } else {
        WriteState::Open
    };

    match mem::replace(&mut transmit.write, new_state) {
        // If a guest is waiting to write, ensure that the next write
        // reflects the closed state of the stream, with
        WriteState::GuestReady {
            ty,
            instance,
            handle,
            close,
            caller,
            err_ctx,
            ..
        } => unsafe {
            push_event(
                store.as_context_mut(),
                transmit_id.rep(),
                match ty {
                    TableIndex::Future(_) => Event::FutureRead,
                    TableIndex::Stream(_) => Event::StreamRead,
                },
                // When closing a reader if the last write was a closing write
                // we must propagate error context if present
                if close {
                    CLOSED | err_ctx as usize
                } else {
                    CLOSED
                },
                caller,
            );

            if close {
                store.concurrent_state().table.delete(transmit_id)?;
            } else {
                *get_mut_by_index(&mut *instance.as_ptr(), ty, handle)?.1 =
                    StreamFutureState::Write;
            }
        },

        // If the host is ready we can receive and discard the write
        WriteState::HostReady { accept, close, .. } => {
            // (????) We can always throw away writers when closing readers because the host should never
            // close a reader *before* a writer.
            accept(Reader::None)?;
            if close {
                store.concurrent_state().table.delete(transmit_id)?;
            }
        }

        WriteState::Open => {}

        WriteState::Closed(_) => {
            log::trace!("host_close_reader delete {transmit_rep}");
            store.concurrent_state().table.delete(transmit_id)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlatAbi {
    size: u32,
    align: u32,
}

/// Represents the writable end of a Component Model `future`.
pub struct FutureWriter<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> FutureWriter<T> {
    /// Write the specified value to this `future`.
    pub fn write<U, S: AsContextMut<Data = U>>(self, store: S, value: T) -> Result<Promise<()>>
    where
        T: func::Lower + Send + Sync + 'static,
    {
        Ok(Promise(Box::pin(
            host_write(store, self.rep, vec![value], true, 0)?.map(drop),
        )))
    }

    /// Close this object without writing a value.
    ///
    /// If this object is dropped without calling either this method or `write`,
    /// any read on the readable end will remain pending forever.
    ///
    /// # Arguments
    ///
    /// * `store` - the store associated with the component instance
    /// * `err_ctx` - the handle of an error context that should be reported with the stream closure (`0` if none)
    ///
    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S, err_ctx: u32) -> Result<()> {
        host_close_writer(store, self.rep, err_ctx)
    }
}

/// Represents the readable end of a Component Model `future`.
pub struct FutureReader<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> FutureReader<T> {
    pub(crate) fn new(rep: u32) -> Self {
        Self {
            rep,
            _phantom: PhantomData,
        }
    }

    /// Read the value from this `future`.
    pub fn read<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<Promise<Option<T>>>
    where
        T: func::Lift + Sync + Send + 'static,
    {
        Ok(Promise(Box::pin(host_read(store, self.rep)?.map(|v| {
            v.ok()
                .and_then(|v| v.map(|v| v.into_iter().next().unwrap()))
        }))))
    }

    /// Convert this `FutureReader` into a [`Val`].
    pub fn into_val(self) -> Val {
        Val::Future(FutureAny(self.rep))
    }

    /// Attempt to convert the specified [`Val`] to a `FutureReader`.
    pub fn from_val<U, S: AsContextMut<Data = U>>(mut store: S, value: &Val) -> Result<Self> {
        let Val::Future(FutureAny(rep)) = value else {
            bail!("expected `future`; got `{}`", value.desc());
        };
        store
            .as_context_mut()
            .concurrent_state()
            .table
            .get(TableId::<TransmitState>::new(*rep))?;
        Ok(Self::new(*rep))
    }

    fn lower_to_index<U>(&self, cx: &mut LowerContext<'_, U>, ty: InterfaceType) -> Result<u32> {
        match ty {
            InterfaceType::Future(dst) => {
                state_table(unsafe { &mut *cx.instance }, TableIndex::Future(dst)).insert(
                    self.rep,
                    WaitableState::Future(dst, StreamFutureState::Read),
                )
            }
            _ => func::bad_type_info(),
        }
    }

    fn lift_from_index(cx: &mut LiftContext<'_>, ty: InterfaceType, index: u32) -> Result<Self> {
        match ty {
            InterfaceType::Future(src) => {
                let state_table =
                    state_table(unsafe { &mut *cx.instance }, TableIndex::Future(src));
                let (rep, state) =
                    get_mut_by_index_from(state_table, TableIndex::Future(src), index)?;

                match state {
                    StreamFutureState::Local => {
                        *state = StreamFutureState::Write;
                    }
                    StreamFutureState::Read => {
                        state_table.remove_by_index(index)?;
                    }
                    StreamFutureState::Write => bail!("cannot transfer write end of future"),
                    StreamFutureState::Busy => bail!("cannot transfer busy future"),
                }

                Ok(Self {
                    rep,
                    _phantom: PhantomData,
                })
            }
            _ => func::bad_type_info(),
        }
    }

    /// Close this object without reading the value.
    ///
    /// If this object is dropped without calling either this method or `read`,
    /// any write on the writable end will remain pending forever.
    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        host_close_reader(store, self.rep)
    }
}

unsafe impl<T> func::ComponentType for FutureReader<T> {
    const ABI: CanonicalAbiInfo = CanonicalAbiInfo::SCALAR4;

    type Lower = <u32 as func::ComponentType>::Lower;

    fn typecheck(ty: &InterfaceType, _types: &InstanceType<'_>) -> Result<()> {
        match ty {
            InterfaceType::Future(_) => Ok(()),
            other => bail!("expected `future`, found `{}`", func::desc(other)),
        }
    }
}

unsafe impl<T> func::Lower for FutureReader<T> {
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

unsafe impl<T> func::Lift for FutureReader<T> {
    fn lift(cx: &mut LiftContext<'_>, ty: InterfaceType, src: &Self::Lower) -> Result<Self> {
        let index = u32::lift(cx, InterfaceType::U32, src)?;
        Self::lift_from_index(cx, ty, index)
    }

    fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Self> {
        let index = u32::load(cx, InterfaceType::U32, bytes)?;
        Self::lift_from_index(cx, ty, index)
    }
}

/// Create a new Component Model `future` as pair of writable and readable ends,
/// the latter of which may be passed to guest code.
pub fn future<T, U, S: AsContextMut<Data = U>>(
    mut store: S,
) -> Result<(FutureWriter<T>, FutureReader<T>)> {
    let mut store = store.as_context_mut();
    let transmit = store.concurrent_state().table.push(TransmitState {
        read: ReadState::Open,
        write: WriteState::Open,
    })?;

    Ok((
        FutureWriter {
            rep: transmit.rep(),
            _phantom: PhantomData,
        },
        FutureReader {
            rep: transmit.rep(),
            _phantom: PhantomData,
        },
    ))
}

/// Represents the writable end of a Component Model `stream`.
pub struct StreamWriter<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> StreamWriter<T> {
    /// Write the specified values to the `stream`.
    pub fn write<U, S: AsContextMut<Data = U>>(
        self,
        store: S,
        values: Vec<T>,
    ) -> Result<Promise<StreamWriter<T>>>
    where
        T: func::Lower + Send + Sync + 'static,
    {
        Ok(Promise(Box::pin(
            host_write(store, self.rep, values, false, 0)?.map(move |_| self),
        )))
    }

    /// Close this object without writing any more values.
    ///
    /// If this object is dropped without calling this method, any read on the
    /// readable end will remain pending forever.
    ///
    /// # Arguments
    ///
    /// * `store` - the store associated with the component instance
    /// * `err_ctx` - the handle of an error context that should be reported with the stream closure (`0` if none)
    ///
    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S, err_ctx: u32) -> Result<()> {
        host_close_writer(store, self.rep, err_ctx)
    }
}

/// Represents the readable end of a Component Model `stream`.
pub struct StreamReader<T> {
    rep: u32,
    _phantom: PhantomData<T>,
}

impl<T> StreamReader<T> {
    pub(crate) fn new(rep: u32) -> Self {
        Self {
            rep,
            _phantom: PhantomData,
        }
    }

    /// Read the next values (if any) from this `stream`.
    pub fn read<U, S: AsContextMut<Data = U>>(
        self,
        store: S,
    ) -> Result<Promise<Option<(StreamReader<T>, Vec<T>)>>>
    where
        T: func::Lift + Sync + Send + 'static,
    {
        Ok(Promise(Box::pin(
            host_read(store, self.rep)?.map(move |v| v.ok().and_then(|v| v.map(|v| (self, v)))),
        )))
    }

    /// Convert this `StreamReader` into a [`Val`].
    pub fn into_val(self) -> Val {
        Val::Stream(StreamAny(self.rep))
    }

    /// Attempt to convert the specified [`Val`] to a `StreamReader`.
    pub fn from_val<U, S: AsContextMut<Data = U>>(mut store: S, value: &Val) -> Result<Self> {
        let Val::Stream(StreamAny(rep)) = value else {
            bail!("expected `stream`; got `{}`", value.desc());
        };
        store
            .as_context_mut()
            .concurrent_state()
            .table
            .get(TableId::<TransmitState>::new(*rep))?;
        Ok(Self::new(*rep))
    }

    fn lower_to_index<U>(&self, cx: &mut LowerContext<'_, U>, ty: InterfaceType) -> Result<u32> {
        match ty {
            InterfaceType::Stream(dst) => {
                state_table(unsafe { &mut *cx.instance }, TableIndex::Stream(dst)).insert(
                    self.rep,
                    WaitableState::Stream(dst, StreamFutureState::Read),
                )
            }
            _ => func::bad_type_info(),
        }
    }

    fn lift_from_index(cx: &mut LiftContext<'_>, ty: InterfaceType, index: u32) -> Result<Self> {
        match ty {
            InterfaceType::Stream(src) => {
                let state_table =
                    state_table(unsafe { &mut *cx.instance }, TableIndex::Stream(src));
                let (rep, state) =
                    get_mut_by_index_from(state_table, TableIndex::Stream(src), index)?;

                match state {
                    StreamFutureState::Local => {
                        *state = StreamFutureState::Write;
                    }
                    StreamFutureState::Read => {
                        state_table.remove_by_index(index)?;
                    }
                    StreamFutureState::Write => bail!("cannot transfer write end of stream"),
                    StreamFutureState::Busy => bail!("cannot transfer busy stream"),
                }

                Ok(Self {
                    rep,
                    _phantom: PhantomData,
                })
            }
            _ => func::bad_type_info(),
        }
    }

    /// Close this object without reading any more values.
    ///
    /// If the object is dropped without either calling this method or reading
    /// until the end of the stream, any write on the writable end will remain
    /// pending forever.
    pub fn close<U, S: AsContextMut<Data = U>>(self, store: S) -> Result<()> {
        host_close_reader(store, self.rep)
    }
}

unsafe impl<T> func::ComponentType for StreamReader<T> {
    const ABI: CanonicalAbiInfo = CanonicalAbiInfo::SCALAR4;

    type Lower = <u32 as func::ComponentType>::Lower;

    fn typecheck(ty: &InterfaceType, _types: &InstanceType<'_>) -> Result<()> {
        match ty {
            InterfaceType::Stream(_) => Ok(()),
            other => bail!("expected `stream`, found `{}`", func::desc(other)),
        }
    }
}

unsafe impl<T> func::Lower for StreamReader<T> {
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

unsafe impl<T> func::Lift for StreamReader<T> {
    fn lift(cx: &mut LiftContext<'_>, ty: InterfaceType, src: &Self::Lower) -> Result<Self> {
        let index = u32::lift(cx, InterfaceType::U32, src)?;
        Self::lift_from_index(cx, ty, index)
    }

    fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Self> {
        let index = u32::load(cx, InterfaceType::U32, bytes)?;
        Self::lift_from_index(cx, ty, index)
    }
}

/// Create a new Component Model `stream` as pair of writable and readable ends,
/// the latter of which may be passed to guest code.
pub fn stream<T, U, S: AsContextMut<Data = U>>(
    mut store: S,
) -> Result<(StreamWriter<T>, StreamReader<T>)> {
    let mut store = store.as_context_mut();
    let transmit = store.concurrent_state().table.push(TransmitState {
        read: ReadState::Open,
        write: WriteState::Open,
    })?;

    Ok((
        StreamWriter {
            rep: transmit.rep(),
            _phantom: PhantomData,
        },
        StreamReader {
            rep: transmit.rep(),
            _phantom: PhantomData,
        },
    ))
}

/// Represents a Component Model `error-context`.
pub struct ErrorContext {
    rep: u32,
}

impl ErrorContext {
    pub(crate) fn new(rep: u32) -> Self {
        Self { rep }
    }

    /// Convert this `ErrorContext` into a [`Val`].
    pub fn into_val(self) -> Val {
        Val::ErrorContext(ErrorContextAny(self.rep))
    }

    /// Attempt to convert the specified [`Val`] to a `ErrorContext`.
    pub fn from_val<U, S: AsContextMut<Data = U>>(_: S, value: &Val) -> Result<Self> {
        let Val::ErrorContext(ErrorContextAny(rep)) = value else {
            bail!("expected `error-context`; got `{}`", value.desc());
        };
        Ok(Self::new(*rep))
    }

    fn lower_to_index<U>(&self, cx: &mut LowerContext<'_, U>, ty: InterfaceType) -> Result<u32> {
        match ty {
            InterfaceType::ErrorContext(dst) => {
                let tbl = unsafe {
                    &mut (*cx.instance)
                        .component_error_context_tables()
                        .get_mut(dst)
                        .expect("error context table index present in (sub)component table during lower")
                };

                if let Some((dst_idx, dst_state)) = tbl.get_mut_by_rep(self.rep) {
                    dst_state.0 += 1;
                    Ok(dst_idx)
                } else {
                    tbl.insert(self.rep, LocalErrorContextRefCount(1))
                }
            }
            _ => func::bad_type_info(),
        }
    }

    fn lift_from_index(cx: &mut LiftContext<'_>, ty: InterfaceType, index: u32) -> Result<Self> {
        match ty {
            InterfaceType::ErrorContext(src) => {
                let (rep, _) = unsafe {
                    (*cx.instance)
                        .component_error_context_tables()
                        .get_mut(src)
                        .expect(
                            "error context table index present in (sub)component table during lift",
                        )
                        .get_mut_by_index(index)?
                };

                Ok(Self { rep })
            }
            _ => func::bad_type_info(),
        }
    }
}

unsafe impl func::ComponentType for ErrorContext {
    const ABI: CanonicalAbiInfo = CanonicalAbiInfo::SCALAR4;

    type Lower = <u32 as func::ComponentType>::Lower;

    fn typecheck(ty: &InterfaceType, _types: &InstanceType<'_>) -> Result<()> {
        match ty {
            InterfaceType::ErrorContext(_) => Ok(()),
            other => bail!("expected `error`, found `{}`", func::desc(other)),
        }
    }
}

unsafe impl func::Lower for ErrorContext {
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

unsafe impl func::Lift for ErrorContext {
    fn lift(cx: &mut LiftContext<'_>, ty: InterfaceType, src: &Self::Lower) -> Result<Self> {
        let index = u32::lift(cx, InterfaceType::U32, src)?;
        Self::lift_from_index(cx, ty, index)
    }

    fn load(cx: &mut LiftContext<'_>, ty: InterfaceType, bytes: &[u8]) -> Result<Self> {
        let index = u32::load(cx, InterfaceType::U32, bytes)?;
        Self::lift_from_index(cx, ty, index)
    }
}

pub(super) struct TransmitState {
    write: WriteState,
    read: ReadState,
}

enum WriteState {
    Open,
    GuestReady {
        ty: TableIndex,
        flat_abi: Option<FlatAbi>,
        options: Options,
        address: usize,
        count: usize,
        instance: SendSyncPtr<ComponentInstance>,
        handle: u32,
        caller: TableId<GuestTask>,
        close: bool,
        /// Error context that may have been written along with the writes
        ///
        /// If the guest wrote This value is zero when there is no error context sent along with the write
        err_ctx: u32,
    },
    HostReady {
        accept: Box<dyn FnOnce(Reader) -> Result<usize> + Send + Sync>,
        close: bool,
        err_ctx: u32,
    },
    /// When write streams are closed, they maybe closed
    /// with an error context that should be used
    ///
    /// Note that this error context is identical to the rep
    /// for the component-global error context table.
    Closed(u32),
}

/// Read state of a transmit channel
///
/// Channels generally start as open, and once they are read for data by either
/// a guest or host, we transition into `GuestReady` or `HostReady` respectively.
///
/// Once a transmit channel is closed, it should *stay* closed.
enum ReadState {
    Open,
    GuestReady {
        ty: TableIndex,
        flat_abi: Option<FlatAbi>,
        options: Options,
        address: usize,
        count: usize,
        instance: SendSyncPtr<ComponentInstance>,
        handle: u32,
        caller: TableId<GuestTask>,
    },
    HostReady {
        accept: Box<dyn FnOnce(Writer) -> Result<usize> + Send + Sync>,
    },
    /// Closed read end, with an optional error context
    ///
    /// If a read operation became closed after a write with an
    /// error context, the final read should receive the error context
    /// as an extra value.
    Closed(u32),
}

enum Writer<'a> {
    /// Writes that are queued from guests
    Guest {
        lift: &'a mut LiftContext<'a>,
        ty: Option<InterfaceType>,
        address: usize,
        count: usize,
        /// Error context that may have been written along with the writes
        ///
        /// If the guest wrote This value is zero when there is no error context sent along with the write
        err_ctx: u32,
    },
    Host {
        values: Box<dyn Any>,
        /// An error context that may have been written along with the given values
        err_ctx: u32,
    },
    None,
}

struct RawLowerContext<'a> {
    store: *mut dyn VMStore,
    options: &'a Options,
    types: &'a Arc<ComponentTypes>,
    instance: *mut ComponentInstance,
}

enum Reader<'a> {
    Guest {
        lower: RawLowerContext<'a>,
        ty: TableIndex,
        address: usize,
        count: usize,
    },
    Host {
        accept: Box<dyn FnOnce(Box<dyn Any>) -> Result<()>>,
    },
    None,
}

/// Create a new waitable state (i.e. for a future or stream)
fn guest_new<T>(vmctx: *mut VMOpaqueContext, ty: TableIndex) -> u64 {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let transmit = cx.concurrent_state().table.push(TransmitState {
                read: ReadState::Open,
                write: WriteState::Open,
            })?;
            state_table(&mut *instance, ty)
                .insert(transmit.rep(), waitable_state(ty, StreamFutureState::Local))
        })
    }
}

unsafe fn copy<T>(
    mut cx: StoreContextMut<'_, T>,
    types: &Arc<ComponentTypes>,
    instance: *mut ComponentInstance,
    flat_abi: Option<FlatAbi>,
    write_ty: TableIndex,
    write_options: &Options,
    write_address: usize,
    read_ty: TableIndex,
    read_options: &Options,
    read_address: usize,
    count: usize,
    rep: u32,
) -> Result<()> {
    match (write_ty, read_ty) {
        (TableIndex::Future(write_ty), TableIndex::Future(read_ty)) => {
            assert_eq!(count, 1);

            let val = types[types[write_ty].ty]
                .payload
                .map(|ty| {
                    let abi = types.canonical_abi(&ty);
                    // FIXME: needs to read an i64 for memory64
                    if write_address % usize::try_from(abi.align32)? != 0 {
                        bail!("write pointer not aligned");
                    }

                    let lift = &mut LiftContext::new(cx.0, write_options, types, instance);

                    let bytes = lift
                        .memory()
                        .get(write_address..)
                        .and_then(|b| b.get(..usize::try_from(abi.size32).unwrap()))
                        .ok_or_else(|| anyhow::anyhow!("write pointer out of bounds of memory"))?;

                    Val::load(lift, ty, bytes)
                })
                .transpose()?;

            if let Some(val) = val {
                let mut lower =
                    LowerContext::new(cx.as_context_mut(), read_options, types, instance);
                let ty = types[types[read_ty].ty].payload.unwrap();
                let ptr = func::validate_inbounds_dynamic(
                    types.canonical_abi(&ty),
                    lower.as_slice_mut(),
                    &ValRaw::u32(read_address.try_into().unwrap()),
                )?;
                val.store(&mut lower, ty, ptr)?;
            }
        }
        (TableIndex::Stream(write_ty), TableIndex::Stream(read_ty)) => {
            let lift = &mut LiftContext::new(cx.0, write_options, types, instance);
            if let Some(flat_abi) = flat_abi {
                // Fast path memcpy for "flat" (i.e. no pointers or handles) payloads:
                let length_in_bytes = usize::try_from(flat_abi.size).unwrap() * count;
                if length_in_bytes > 0 {
                    if write_address % usize::try_from(flat_abi.align)? != 0 {
                        bail!("write pointer not aligned");
                    }
                    if read_address % usize::try_from(flat_abi.align)? != 0 {
                        bail!("read pointer not aligned");
                    }

                    {
                        let src = write_options
                            .memory(cx.0)
                            .get(write_address..)
                            .and_then(|b| b.get(..length_in_bytes))
                            .ok_or_else(|| {
                                anyhow::anyhow!("write pointer out of bounds of memory")
                            })?
                            .as_ptr();
                        let dst = read_options
                            .memory_mut(cx.0)
                            .get_mut(read_address..)
                            .and_then(|b| b.get_mut(..length_in_bytes))
                            .ok_or_else(|| anyhow::anyhow!("read pointer out of bounds of memory"))?
                            .as_mut_ptr();
                        src.copy_to(dst, length_in_bytes);
                    }
                }
            } else {
                let ty = types[types[write_ty].ty].payload.unwrap();
                let abi = lift.types.canonical_abi(&ty);
                let size = usize::try_from(abi.size32).unwrap();
                if write_address % usize::try_from(abi.align32)? != 0 {
                    bail!("write pointer not aligned");
                }
                let bytes = lift
                    .memory()
                    .get(write_address..)
                    .and_then(|b| b.get(..size * count))
                    .ok_or_else(|| anyhow::anyhow!("write pointer out of bounds of memory"))?;

                let values = (0..count)
                    .map(|index| Val::load(lift, ty, &bytes[(index * size)..][..size]))
                    .collect::<Result<Vec<_>>>()?;

                log::trace!("copy values {values:?} for {rep}");

                let lower =
                    &mut LowerContext::new(cx.as_context_mut(), read_options, types, instance);
                let ty = types[types[read_ty].ty].payload.unwrap();
                let abi = lower.types.canonical_abi(&ty);
                if read_address % usize::try_from(abi.align32)? != 0 {
                    bail!("read pointer not aligned");
                }
                let size = usize::try_from(abi.size32).unwrap();
                lower
                    .as_slice_mut()
                    .get_mut(read_address..)
                    .and_then(|b| b.get_mut(..size * count))
                    .ok_or_else(|| anyhow::anyhow!("read pointer out of bounds of memory"))?;
                let mut ptr = read_address;
                for value in values {
                    value.store(lower, ty, ptr)?;
                    ptr += size
                }
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}

fn guest_write<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TableIndex,
    flat_abi: Option<FlatAbi>,
    handle: u32,
    address: u32,
    count: u32,
) -> u64 {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let address = usize::try_from(address).unwrap();
            let count = usize::try_from(count).unwrap();
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                NonNull::new(realloc),
                StringEncoding::from_u8(string_encoding).unwrap(),
                true,
                None,
            );
            let types = (*instance).component_types();
            let (rep, state) = get_mut_by_index(&mut *instance, ty, handle)?;
            let StreamFutureState::Write = *state else {
                bail!("invalid handle");
            };
            *state = StreamFutureState::Busy;
            let transmit_id = TableId::<TransmitState>::new(rep);
            let transmit = cx.concurrent_state().table.get_mut(transmit_id)?;

            let new_state = if let ReadState::Closed(err_ctx) = &transmit.read {
                ReadState::Closed(*err_ctx)
            } else {
                ReadState::Open
            };

            // TODO: If the read end is in a closed state with an error context,
            // ensure this component has access to the error context.

            let result = match mem::replace(&mut transmit.read, new_state) {
                // If the read state represents a guest that is waiting to read,
                // we can continue with our write
                ReadState::GuestReady {
                    ty: read_ty,
                    flat_abi: read_flat_abi,
                    options: read_options,
                    address: read_address,
                    count: read_count,
                    instance: _,
                    handle: read_handle,
                    caller: read_caller,
                } => {
                    assert_eq!(flat_abi, read_flat_abi);

                    let count = count.min(read_count);

                    copy(
                        cx.as_context_mut(),
                        types,
                        instance,
                        flat_abi,
                        ty,
                        &options,
                        address,
                        read_ty,
                        &read_options,
                        read_address,
                        count,
                        rep,
                    )?;

                    log::trace!(
                        "remove read child of {}: {}",
                        read_caller.rep(),
                        transmit_id.rep()
                    );
                    cx.concurrent_state()
                        .table
                        .remove_child(transmit_id, read_caller)?;

                    *get_mut_by_index(&mut *instance, read_ty, read_handle)?.1 =
                        StreamFutureState::Read;

                    push_event(
                        cx,
                        transmit_id.rep(),
                        match read_ty {
                            TableIndex::Future(_) => Event::FutureRead,
                            TableIndex::Stream(_) => Event::StreamRead,
                        },
                        count,
                        read_caller,
                    );

                    count
                }

                // If the read state represents the host being ready to read, we can perform the write
                // against the callback left by the host for accepting the read.
                ReadState::HostReady { accept } => {
                    let lift = &mut LiftContext::new(cx.0, &options, types, instance);

                    // TODO: during guest write,

                    accept(Writer::Guest {
                        lift,
                        ty: payload(ty, types),
                        address,
                        count,
                        // TODO: do we need this here?
                        err_ctx: 0,
                    })?
                }

                // If the read state indicates that no waiters have yet come along interested in the value
                // we save the guest's intent to write
                ReadState::Open => {
                    assert!(matches!(&transmit.write, WriteState::Open));

                    let caller = cx.concurrent_state().guest_task.unwrap();
                    log::trace!(
                        "add write {} child of {}: {}",
                        match ty {
                            TableIndex::Future(_) => "future",
                            TableIndex::Stream(_) => "stream",
                        },
                        caller.rep(),
                        transmit_id.rep()
                    );
                    cx.concurrent_state().table.add_child(transmit_id, caller)?;

                    let transmit = cx.concurrent_state().table.get_mut(transmit_id)?;
                    transmit.write = WriteState::GuestReady {
                        ty,
                        flat_abi,
                        options,
                        address: usize::try_from(address).unwrap(),
                        count: usize::try_from(count).unwrap(),
                        instance: SendSyncPtr::new(NonNull::new(instance).unwrap()),
                        handle,
                        caller,
                        close: false,
                        err_ctx: 0,
                    };

                    BLOCKED
                }

                // If we receive a closed read state, we return ensure that the current task
                // has the relevant error context available locally, and return the indicator
                // of task closed with the error context handle
                ReadState::Closed(err_ctx) => {
                    // Look up existing global err_ctx
                    ensure!(cx
                        .concurrent_state()
                        .table
                        .get(TableId::<ErrorContextState>::new(err_ctx))
                        .is_ok());
                    // Add the global error context to this local component instance if it's not present already
                    let local_tbl_id = TypeComponentLocalErrorContextTableIndex::from_u32(err_ctx);
                    let local_tbl = (*instance)
                        .component_error_context_tables()
                        .get_mut_or_insert_with(local_tbl_id, || StateTable::default());
                    if local_tbl.has_handle(local_tbl_id.as_u32()) {
                        let (_, LocalErrorContextRefCount(ref mut n)) =
                            local_tbl.get_mut_by_index(local_tbl_id.as_u32())?;
                        *n += 1;
                    } else {
                        local_tbl
                            .insert(local_tbl_id.as_u32(), LocalErrorContextRefCount(1))
                            .context("copying local error context during closing guest write")?;
                    }

                    CLOSED | err_ctx as usize
                }
            };

            if result != BLOCKED {
                *get_mut_by_index(&mut *instance, ty, handle)?.1 = StreamFutureState::Write;
            }

            Ok(u32::try_from(result).unwrap())
        })
    }
}

fn guest_read<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TableIndex,
    flat_abi: Option<FlatAbi>,
    handle: u32,
    address: u32,
    count: u32,
) -> u64 {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let address = usize::try_from(address).unwrap();
            let count = usize::try_from(count).unwrap();
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                NonNull::new(realloc),
                StringEncoding::from_u8(string_encoding).unwrap(),
                true,
                None,
            );
            let types = (*instance).component_types();
            let (rep, state) = get_mut_by_index(&mut *instance, ty, handle)?;
            let StreamFutureState::Read = *state else {
                bail!("invalid handle");
            };
            *state = StreamFutureState::Busy;
            let transmit_id = TableId::<TransmitState>::new(rep);
            let transmit = cx.concurrent_state().table.get_mut(transmit_id)?;

            // Get the current write status
            let (new_state, err_ctx) = if let WriteState::Closed(err_ctx) = &transmit.write {
                (WriteState::Closed(*err_ctx), *err_ctx)
            } else {
                (WriteState::Open, 0)
            };

            let result = match mem::replace(&mut transmit.write, new_state) {
                WriteState::GuestReady {
                    ty: write_ty,
                    flat_abi: write_flat_abi,
                    options: write_options,
                    address: write_address,
                    count: write_count,
                    instance: _,
                    handle: write_handle,
                    caller: write_caller,
                    close,
                    err_ctx,
                } => {
                    assert_eq!(flat_abi, write_flat_abi);

                    let count = count.min(write_count);

                    copy(
                        cx.as_context_mut(),
                        types,
                        instance,
                        flat_abi,
                        write_ty,
                        &write_options,
                        write_address,
                        ty,
                        &options,
                        address,
                        count,
                        rep,
                    )?;

                    log::trace!(
                        "remove write child of {}: {}",
                        write_caller.rep(),
                        transmit_id.rep()
                    );
                    cx.concurrent_state()
                        .table
                        .remove_child(transmit_id, write_caller)?;

                    // If the writer elected to close the channel after writing, pass along the error context
                    if close {
                        cx.concurrent_state().table.get_mut(transmit_id)?.write =
                            WriteState::Closed(err_ctx);
                    } else {
                        *get_mut_by_index(&mut *instance, write_ty, write_handle)?.1 =
                            StreamFutureState::Write;
                    }

                    push_event(
                        cx,
                        transmit_id.rep(),
                        match write_ty {
                            TableIndex::Future(_) => Event::FutureWrite,
                            TableIndex::Stream(_) => Event::StreamWrite,
                        },
                        count,
                        write_caller,
                    );

                    count
                }

                WriteState::HostReady {
                    accept,
                    close,
                    err_ctx,
                } => {
                    let count = accept(Reader::Guest {
                        lower: RawLowerContext {
                            store: cx.0.traitobj(),
                            options: &options,
                            types,
                            instance,
                        },
                        ty,
                        address: usize::try_from(address).unwrap(),
                        count,
                    })?;

                    // If the host writer elected to close the channel after writing, pass along the error context
                    if close {
                        cx.concurrent_state().table.get_mut(transmit_id)?.write =
                            WriteState::Closed(err_ctx);
                    }

                    count
                }

                WriteState::Open => {
                    assert!(matches!(&transmit.read, ReadState::Open));

                    let caller = cx.concurrent_state().guest_task.unwrap();
                    log::trace!(
                        "add read {} child of {}: {}",
                        match ty {
                            TableIndex::Future(_) => "future",
                            TableIndex::Stream(_) => "stream",
                        },
                        caller.rep(),
                        transmit_id.rep()
                    );
                    cx.concurrent_state().table.add_child(transmit_id, caller)?;

                    let transmit = cx.concurrent_state().table.get_mut(transmit_id)?;
                    transmit.read = ReadState::GuestReady {
                        ty,
                        flat_abi,
                        options,
                        address: usize::try_from(address).unwrap(),
                        count: usize::try_from(count).unwrap(),
                        instance: SendSyncPtr::new(NonNull::new(instance).unwrap()),
                        handle,
                        caller,
                    };

                    BLOCKED
                }

                // If at some point the writer chose to close, the final stream that comes back
                // should contain CLOSED and the error context
                WriteState::Closed(err_ctx) => CLOSED | err_ctx as usize,
            };

            if result != BLOCKED {
                *get_mut_by_index(&mut *instance, ty, handle)?.1 = StreamFutureState::Read;
            }

            Ok(u32::try_from(result).unwrap())
        })
    }
}

fn guest_cancel_write<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TableIndex,
    writer: u32,
    _async_: bool,
) -> u64 {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let (rep, WaitableState::Stream(_, state) | WaitableState::Future(_, state)) =
                state_table(&mut *instance, ty).get_mut_by_index(writer)?
            else {
                bail!("invalid stream or future handle");
            };
            match state {
                StreamFutureState::Local | StreamFutureState::Write => {
                    bail!("stream or future write canceled when no write is pending")
                }
                StreamFutureState::Read => {
                    bail!("passed read end to `{{stream|future}}.cancel-write`")
                }
                StreamFutureState::Busy => {
                    *state = StreamFutureState::Write;
                }
            }
            host_cancel_write(cx, rep)
        })
    }
}

fn guest_cancel_read<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TableIndex,
    reader: u32,
    _async_: bool,
) -> u64 {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let (rep, WaitableState::Stream(_, state) | WaitableState::Future(_, state)) =
                state_table(&mut *instance, ty).get_mut_by_index(reader)?
            else {
                bail!("invalid stream or future handle");
            };
            match state {
                StreamFutureState::Local | StreamFutureState::Read => {
                    bail!("stream or future read canceled when no read is pending")
                }
                StreamFutureState::Write => {
                    bail!("passed write end to `{{stream|future}}.cancel-read`")
                }
                StreamFutureState::Busy => {
                    *state = StreamFutureState::Read;
                }
            }
            host_cancel_read(cx, rep)
        })
    }
}

fn guest_close_writable<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TableIndex,
    writer: u32,
    err_ctx: u32,
) -> bool {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());

            let (rep, WaitableState::Stream(_, state) | WaitableState::Future(_, state)) =
                state_table(&mut *instance, ty).remove_by_index(writer)?
            else {
                bail!("invalid stream or future handle");
            };
            match state {
                StreamFutureState::Local | StreamFutureState::Write => {}
                StreamFutureState::Read => {
                    bail!("passed read end to `{{stream|future}}.close-writable`")
                }
                StreamFutureState::Busy => bail!("cannot drop busy stream or future"),
            }

            // If an error context was provided, ensure it's valid
            if err_ctx != 0 {
                assert!(
                    (*instance)
                        .component_error_context_tables()
                        .get(TypeComponentLocalErrorContextTableIndex::from_u32(err_ctx))
                        .is_some(),
                    "invalid component-local error context handle"
                );
                assert!(
                    cx.concurrent_state()
                        .table
                        .get(TableId::<ErrorContextState>::new(err_ctx))
                        .is_ok(),
                    "failed to find error context state"
                );
            }

            host_close_writer(cx, rep, err_ctx)
        })
    }
}

fn guest_close_readable<T>(vmctx: *mut VMOpaqueContext, ty: TableIndex, reader: u32) -> bool {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let (rep, WaitableState::Stream(_, state) | WaitableState::Future(_, state)) =
                state_table(&mut *instance, ty).remove_by_index(reader)?
            else {
                bail!("invalid stream or future handle");
            };
            match state {
                StreamFutureState::Local | StreamFutureState::Read => {}
                StreamFutureState::Write => {
                    bail!("passed write end to `{{stream|future}}.close-readable`")
                }
                StreamFutureState::Busy => bail!("cannot drop busy stream or future"),
            }
            host_close_reader(cx, rep)
        })
    }
}

pub(crate) extern "C" fn future_new<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
) -> u64 {
    guest_new::<T>(vmctx, TableIndex::Future(ty))
}

pub(crate) extern "C" fn future_write<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TypeFutureTableIndex,
    future: u32,
    address: u32,
) -> u64 {
    guest_write::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Future(ty),
        None,
        future,
        address,
        1,
    )
}

pub(crate) extern "C" fn future_read<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TypeFutureTableIndex,
    future: u32,
    address: u32,
) -> u64 {
    guest_read::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Future(ty),
        None,
        future,
        address,
        1,
    )
}

pub(crate) extern "C" fn future_cancel_write<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    async_: bool,
    writer: u32,
) -> u64 {
    guest_cancel_write::<T>(vmctx, TableIndex::Future(ty), writer, async_)
}

pub(crate) extern "C" fn future_cancel_read<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    async_: bool,
    reader: u32,
) -> u64 {
    guest_cancel_read::<T>(vmctx, TableIndex::Future(ty), reader, async_)
}

pub(crate) extern "C" fn future_close_writable<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    writer: u32,
    error: u32,
) -> bool {
    guest_close_writable::<T>(vmctx, TableIndex::Future(ty), writer, error)
}

pub(crate) extern "C" fn future_close_readable<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeFutureTableIndex,
    reader: u32,
) -> bool {
    guest_close_readable::<T>(vmctx, TableIndex::Future(ty), reader)
}

pub(crate) extern "C" fn stream_new<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
) -> u64 {
    guest_new::<T>(vmctx, TableIndex::Stream(ty))
}

pub(crate) extern "C" fn stream_write<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TypeStreamTableIndex,
    stream: u32,
    address: u32,
    count: u32,
) -> u64 {
    guest_write::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Stream(ty),
        None,
        stream,
        address,
        count,
    )
}

pub(crate) extern "C" fn stream_read<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TypeStreamTableIndex,
    stream: u32,
    address: u32,
    count: u32,
) -> u64 {
    guest_read::<T>(
        vmctx,
        memory,
        realloc,
        string_encoding,
        TableIndex::Stream(ty),
        None,
        stream,
        address,
        count,
    )
}

pub(crate) extern "C" fn stream_cancel_write<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    async_: bool,
    writer: u32,
) -> u64 {
    guest_cancel_write::<T>(vmctx, TableIndex::Stream(ty), writer, async_)
}

pub(crate) extern "C" fn stream_cancel_read<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    async_: bool,
    reader: u32,
) -> u64 {
    guest_cancel_read::<T>(vmctx, TableIndex::Stream(ty), reader, async_)
}

pub(crate) extern "C" fn stream_close_writable<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    writer: u32,
    error: u32,
) -> bool {
    guest_close_writable::<T>(vmctx, TableIndex::Stream(ty), writer, error)
}

pub(crate) extern "C" fn stream_close_readable<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeStreamTableIndex,
    reader: u32,
) -> bool {
    guest_close_readable::<T>(vmctx, TableIndex::Stream(ty), reader)
}

pub(crate) extern "C" fn flat_stream_write<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    ty: TypeStreamTableIndex,
    payload_size: u32,
    payload_align: u32,
    stream: u32,
    address: u32,
    count: u32,
) -> u64 {
    guest_write::<T>(
        vmctx,
        memory,
        realloc,
        StringEncoding::Utf8 as u8,
        TableIndex::Stream(ty),
        Some(FlatAbi {
            size: payload_size,
            align: payload_align,
        }),
        stream,
        address,
        count,
    )
}

pub(crate) extern "C" fn flat_stream_read<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    ty: TypeStreamTableIndex,
    payload_size: u32,
    payload_align: u32,
    stream: u32,
    address: u32,
    count: u32,
) -> u64 {
    guest_read::<T>(
        vmctx,
        memory,
        realloc,
        StringEncoding::Utf8 as u8,
        TableIndex::Stream(ty),
        Some(FlatAbi {
            size: payload_size,
            align: payload_align,
        }),
        stream,
        address,
        count,
    )
}

/// Create a new error context for the given component
pub(crate) extern "C" fn error_context_new<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TypeComponentLocalErrorContextTableIndex,
    debug_msg_address: u32,
    debug_msg_len: u32,
) -> u64 {
    unsafe {
        call_host_and_handle_result::<T, u32>(vmctx, || {
            // Retrieve the component instance
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();

            //  Read string from guest memory
            let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let options = Options::new(
                cx.0.id(),
                NonNull::new(memory),
                NonNull::new(realloc),
                StringEncoding::from_u8(string_encoding).ok_or_else(|| {
                    anyhow::anyhow!("failed to convert u8 string encoding [{string_encoding}]")
                })?,
                false,
                None,
            );
            let lift_ctx =
                &mut LiftContext::new(cx.0, &options, (*instance).component_types(), instance);
            let s = {
                let address = usize::try_from(debug_msg_address)?;
                let len = usize::try_from(debug_msg_len)?;
                WasmStr::load(
                    lift_ctx,
                    InterfaceType::String,
                    &lift_ctx
                        .memory()
                        .get(address..)
                        .and_then(|b| b.get(..len))
                        .map(|_| {
                            [debug_msg_address.to_le_bytes(), debug_msg_len.to_le_bytes()].concat()
                        })
                        .ok_or_else(|| {
                            anyhow::anyhow!("invalid debug message pointer: out of bounds")
                        })?,
                )?
            };

            // Create a new ErrorContext that is tracked along with other concurrent state
            let err_ctx = ErrorContextState {
                debug_msg: s.to_str(&cx)?.to_string(),
            };
            let table_id = cx.concurrent_state().table.push(err_ctx)?;
            let global_ref_count_idx =
                TypeComponentGlobalErrorContextTableIndex::from_u32(table_id.rep());

            // Add to the global error context ref counts
            let _ = (*instance)
                .component_global_error_context_ref_counts()
                .insert(global_ref_count_idx, GlobalErrorContextRefCount(1));

            // Error context are tracked both locally (to a single component instance) and globally
            // the counts for both must stay in sync.
            //
            // Here we reflect the newly created global concurrent error context state into the
            // component instance's locally tracked count, along with the appropriate key into the global
            // ref tracking data structures to enable later lookup
            let local_tbl = (*instance)
                .component_error_context_tables()
                .get_mut_or_insert_with(ty, || StateTable::default());
            assert!(
                !local_tbl.has_handle(table_id.rep()),
                "newly created error context state already tracked by component"
            );
            let local_idx = local_tbl.insert(table_id.rep(), LocalErrorContextRefCount(1))?;

            Ok(local_idx)
        })
    }
}

pub(crate) extern "C" fn error_context_debug_message<T>(
    vmctx: *mut VMOpaqueContext,
    memory: *mut VMMemoryDefinition,
    realloc: *mut VMFuncRef,
    string_encoding: u8,
    ty: TypeComponentLocalErrorContextTableIndex,
    err_ctx_handle: u32,
    debug_msg_address: u32,
) -> bool {
    unsafe {
        call_host_and_handle_result::<T, ()>(vmctx, || {
            // Retrieve the component instance
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());
            let store_id = cx.0.id();

            // Retrieve the error context and internal debug message
            let (state_table_id_rep, _) = (*instance)
                .component_error_context_tables()
                .get_mut(ty)
                .context(
                    "error context table index present in (sub)component lookup during debug_msg",
                )?
                .get_mut_by_index(err_ctx_handle)?;

            // Get the state associated with the error context
            let ErrorContextState { debug_msg } = cx
                .concurrent_state()
                .table
                .get_mut(TableId::<ErrorContextState>::new(state_table_id_rep))?;
            let debug_msg = debug_msg.clone();

            // Lower the string into the component's memory
            let options = Options::new(
                store_id,
                NonNull::new(memory),
                NonNull::new(realloc),
                StringEncoding::from_u8(string_encoding).ok_or_else(|| {
                    anyhow::anyhow!("failed to convert u8 string encoding [{string_encoding}]")
                })?,
                false,
                None,
            );
            let lower_cx =
                &mut LowerContext::new(cx, &options, (*instance).component_types(), instance);
            let debug_msg_address = usize::try_from(debug_msg_address)?;
            let offset = lower_cx
                .as_slice_mut()
                .get(debug_msg_address..)
                .and_then(|b| b.get(..debug_msg.bytes().len()))
                .map(|_| debug_msg_address)
                .ok_or_else(|| anyhow::anyhow!("invalid debug message pointer: out of bounds"))?;
            debug_msg
                .as_str()
                .store(lower_cx, InterfaceType::String, offset)?;

            Ok(())
        })
    }
}

pub(crate) extern "C" fn error_context_drop<T>(
    vmctx: *mut VMOpaqueContext,
    ty: TypeComponentLocalErrorContextTableIndex,
    error_context: u32,
) -> bool {
    unsafe {
        call_host_and_handle_result::<T, _>(vmctx, || {
            let cx = VMComponentContext::from_opaque(vmctx);
            let instance = (*cx).instance();
            let local_state_table = (*instance)
                .component_error_context_tables()
                .get_mut(ty)
                .context("error context table index present in (sub)component table during drop")?;

            // Reduce the local (sub)component ref count, removing tracking if necessary
            let (rep, local_ref_removed) = {
                let (rep, LocalErrorContextRefCount(local_ref_count)) =
                    local_state_table.get_mut_by_index(error_context)?;
                assert!(*local_ref_count > 0);
                *local_ref_count -= 1;
                let mut local_ref_removed = false;
                if *local_ref_count == 0 {
                    local_ref_removed = true;
                    local_state_table
                        .remove_by_index(error_context)
                        .context("removing error context from component-local tracking")?;
                }
                (rep, local_ref_removed)
            };

            let global_ref_count_idx = TypeComponentGlobalErrorContextTableIndex::from_u32(rep);

            let GlobalErrorContextRefCount(global_ref_count) = (*instance)
                .component_global_error_context_ref_counts()
                .get_mut(&global_ref_count_idx)
                .expect("retrieve concurrent state for error context during drop");

            // Reduce the component-global ref count, removing tracking if necessary
            assert!(*global_ref_count >= 1);
            *global_ref_count -= 1;
            if *global_ref_count == 0 {
                assert!(local_ref_removed);
                let mut cx = StoreContextMut::<T>(&mut *(*instance).store().cast());

                (*instance)
                    .component_global_error_context_ref_counts()
                    .remove(&global_ref_count_idx);

                cx.concurrent_state()
                    .table
                    .delete(TableId::<ErrorContextState>::new(rep))
                    .context("deleting component-global error context data")?;
            }

            Ok(())
        })
    }
}
