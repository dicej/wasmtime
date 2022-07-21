use crate::component::instance::{Instance, InstanceData};
use crate::component::types::{SizeAndAlignment, Type};
use crate::component::values::Val;
use crate::store::{StoreOpaque, Stored};
use crate::{AsContext, AsContextMut, StoreContextMut, ValRaw};
use anyhow::{bail, Context, Result};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;
use wasmtime_environ::component::{
    CanonicalOptions, ComponentTypes, CoreDef, RuntimeComponentInstanceIndex, TypeFuncIndex,
};
use wasmtime_runtime::{Export, ExportFunction, VMTrampoline};

pub(crate) const MAX_STACK_PARAMS: usize = 16;
pub(crate) const MAX_STACK_RESULTS: usize = 1;

/// A helper macro to safely map `MaybeUninit<T>` to `MaybeUninit<U>` where `U`
/// is a field projection within `T`.
///
/// This is intended to be invoked as:
///
/// ```ignore
/// struct MyType {
///     field: u32,
/// }
///
/// let initial: &mut MaybeUninit<MyType> = ...;
/// let field: &mut MaybeUninit<u32> = map_maybe_uninit!(initial.field);
/// ```
///
/// Note that array accesses are also supported:
///
/// ```ignore
///
/// let initial: &mut MaybeUninit<[u32; 2]> = ...;
/// let element: &mut MaybeUninit<u32> = map_maybe_uninit!(initial[1]);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! map_maybe_uninit {
    ($maybe_uninit:ident $($field:tt)*) => (#[allow(unused_unsafe)] unsafe {
        use $crate::component::__internal::MaybeUninitExt;

        let m: &mut std::mem::MaybeUninit<_> = $maybe_uninit;
        // Note the usage of `addr_of_mut!` here which is an attempt to "stay
        // safe" here where we never accidentally create `&mut T` where `T` is
        // actually uninitialized, hopefully appeasing the Rust unsafe
        // guidelines gods.
        m.map(|p| std::ptr::addr_of_mut!((*p)$($field)*))
    })
}

#[doc(hidden)]
pub trait MaybeUninitExt<T> {
    /// Maps `MaybeUninit<T>` to `MaybeUninit<U>` using the closure provided.
    ///
    /// Note that this is `unsafe` as there is no guarantee that `U` comes from
    /// `T`.
    unsafe fn map<U>(&mut self, f: impl FnOnce(*mut T) -> *mut U) -> &mut MaybeUninit<U>;
}

impl<T> MaybeUninitExt<T> for MaybeUninit<T> {
    unsafe fn map<U>(&mut self, f: impl FnOnce(*mut T) -> *mut U) -> &mut MaybeUninit<U> {
        let new_ptr = f(self.as_mut_ptr());
        std::mem::transmute::<*mut U, &mut MaybeUninit<U>>(new_ptr)
    }
}

mod host;
mod options;
mod typed;
pub use self::host::*;
pub use self::options::*;
pub use self::typed::*;

/// A WebAssembly component function.
//
// FIXME: write more docs here
#[derive(Copy, Clone, Debug)]
pub struct Func(Stored<FuncData>);

#[doc(hidden)]
pub struct FuncData {
    trampoline: VMTrampoline,
    export: ExportFunction,
    ty: TypeFuncIndex,
    types: Arc<ComponentTypes>,
    options: Options,
    instance: Instance,
    component_instance: RuntimeComponentInstanceIndex,
    post_return: Option<(ExportFunction, VMTrampoline)>,
    post_return_arg: Option<ValRaw>,
}

impl Func {
    pub(crate) fn from_lifted_func(
        store: &mut StoreOpaque,
        instance: &Instance,
        data: &InstanceData,
        ty: TypeFuncIndex,
        func: &CoreDef,
        options: &CanonicalOptions,
    ) -> Func {
        let export = match data.lookup_def(store, func) {
            Export::Function(f) => f,
            _ => unreachable!(),
        };
        let trampoline = store.lookup_trampoline(unsafe { export.anyfunc.as_ref() });
        let memory = options
            .memory
            .map(|i| NonNull::new(data.instance().runtime_memory(i)).unwrap());
        let realloc = options.realloc.map(|i| data.instance().runtime_realloc(i));
        let post_return = options.post_return.map(|i| {
            let anyfunc = data.instance().runtime_post_return(i);
            let trampoline = store.lookup_trampoline(unsafe { anyfunc.as_ref() });
            (ExportFunction { anyfunc }, trampoline)
        });
        let component_instance = options.instance;
        let options = unsafe { Options::new(store.id(), memory, realloc, options.string_encoding) };
        Func(store.store_data_mut().insert(FuncData {
            trampoline,
            export,
            options,
            ty,
            types: data.component_types().clone(),
            instance: *instance,
            component_instance,
            post_return,
            post_return_arg: None,
        }))
    }

    /// Attempt to cast this [`Func`] to a statically typed [`TypedFunc`] with
    /// the provided `Params` and `Return`.
    ///
    /// This function will perform a type-check at runtime that the [`Func`]
    /// takes `Params` as parameters and returns `Return`. If the type-check
    /// passes then a [`TypedFunc`] will be returned which can be used to
    /// invoke the function in an efficient, statically-typed, and ergonomic
    /// manner.
    ///
    /// The `Params` type parameter here is a tuple of the parameters to the
    /// function. A function which takes no arguments should use `()`, a
    /// function with one argument should use `(T,)`, etc. Note that all
    /// `Params` must also implement the [`Lower`] trait since they're going tin
    /// to wasm.
    ///
    /// The `Return` type parameter is the return value of this function. A
    /// return value of `()` means that there's no return (similar to a Rust
    /// unit return) and otherwise a type `T` can be specified. Note that the
    /// `Return` must also implement the [`Lift`] trait since it's coming from
    /// wasm.
    ///
    /// Types specified here must implement the [`ComponentType`] trait. This
    /// trait is implemented for built-in types to Rust such as integer
    /// primitives, floats, `Option<T>`, `Result<T, E>`, strings, `Vec<T>`, and
    /// more. As parameters you'll be passing native Rust types.
    ///
    /// # Errors
    ///
    /// If the function does not actually take `Params` as its parameters or
    /// return `Return` then an error will be returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `self` is not owned by the `store`
    /// specified.
    ///
    /// # Examples
    ///
    /// Calling a function which takes no parameters and has no return value:
    ///
    /// ```
    /// # use wasmtime::component::Func;
    /// # use wasmtime::Store;
    /// # fn foo(func: &Func, store: &mut Store<()>) -> anyhow::Result<()> {
    /// let typed = func.typed::<(), (), _>(&store)?;
    /// typed.call(store, ())?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Calling a function which takes one string parameter and returns a
    /// string:
    ///
    /// ```
    /// # use wasmtime::component::{Func, Value};
    /// # use wasmtime::Store;
    /// # fn foo(func: &Func, mut store: Store<()>) -> anyhow::Result<()> {
    /// let typed = func.typed::<(&str,), Value<String>, _>(&store)?;
    /// let ret = typed.call(&mut store, ("Hello, ",))?;
    /// let ret = ret.cursor(&store);
    /// println!("returned string was: {}", ret.to_str()?);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Calling a function which takes multiple parameters and returns a boolean:
    ///
    /// ```
    /// # use wasmtime::component::Func;
    /// # use wasmtime::Store;
    /// # fn foo(func: &Func, mut store: Store<()>) -> anyhow::Result<()> {
    /// let typed = func.typed::<(u32, Option<&str>, &[u8]), bool, _>(&store)?;
    /// let ok: bool = typed.call(&mut store, (1, Some("hello"), b"bytes!"))?;
    /// println!("return value was: {ok}");
    /// # Ok(())
    /// # }
    /// ```
    pub fn typed<Params, Return, S>(&self, store: S) -> Result<TypedFunc<Params, Return>>
    where
        Params: ComponentParams + Lower,
        Return: Lift,
        S: AsContext,
    {
        self._typed(store.as_context().0)
    }

    pub(crate) fn _typed<Params, Return>(
        &self,
        store: &StoreOpaque,
    ) -> Result<TypedFunc<Params, Return>>
    where
        Params: ComponentParams + Lower,
        Return: Lift,
    {
        self.typecheck::<Params, Return>(store)?;
        unsafe { Ok(TypedFunc::new_unchecked(*self)) }
    }

    fn typecheck<Params, Return>(&self, store: &StoreOpaque) -> Result<()>
    where
        Params: ComponentParams + Lower,
        Return: Lift,
    {
        let data = &store[self.0];
        let ty = &data.types[data.ty];

        Params::typecheck_params(&ty.params, &data.types)
            .context("type mismatch with parameters")?;
        Return::typecheck(&ty.result, &data.types).context("type mismatch with result")?;

        Ok(())
    }

    /// Get the parameter types for this function.
    pub fn params(&self, store: impl AsContext) -> Box<[Type]> {
        let data = &store.as_context()[self.0];
        data.types[data.ty]
            .params
            .iter()
            .map(|(_, ty)| Type::from(ty, &data.types))
            .collect()
    }

    /// Invokes this function with the `params` given and returns the result.
    ///
    /// The `params` here must match the type signature of this `Func`, or this will return an error. If a trap
    /// occurs while executing this function, then an error will also be returned.
    pub fn call(&self, mut store: impl AsContextMut, args: &[Val]) -> Result<Val> {
        let store = &mut store.as_context_mut();

        let params;
        let result;

        {
            let data = &store[self.0];
            let ty = &data.types[data.ty];

            if ty.params.len() != args.len() {
                bail!(
                    "expected {} argument(s), got {}",
                    ty.params.len(),
                    args.len()
                );
            }

            params = ty
                .params
                .iter()
                .zip(args)
                .map(|((_, ty), arg)| {
                    let ty = Type::from(ty, &data.types);

                    ty.check(arg).context("type mismatch with parameters")?;

                    Ok(ty)
                })
                .collect::<Result<Vec<_>>>()?;

            result = Type::from(&ty.result, &data.types);
        }

        let param_count = params.iter().map(|ty| ty.flatten_count()).sum::<usize>();
        let result_count = result.flatten_count();

        call_raw(
            self,
            store,
            args,
            |store, options, args, dst: &mut MaybeUninit<[ValRaw; MAX_STACK_PARAMS]>| {
                if param_count > MAX_STACK_PARAMS {
                    self.store_args(store, &options, &params, args, dst)
                } else {
                    args.iter()
                        .try_for_each(|arg| arg.lower(store, &options, dst, 0))
                }
            },
            |store, options, src: &[ValRaw; MAX_STACK_RESULTS]| {
                if result_count > MAX_STACK_RESULTS {
                    Self::load_result(
                        store,
                        &Memory::new(store, &options),
                        &result,
                        &mut src.iter(),
                    )
                } else {
                    result.lift(store, &options, &mut src.iter())
                }
            },
        )
    }

    /// Invokes the `post-return` canonical ABI option, if specified, after a
    /// [`Func::call`] has finished.
    ///
    /// For some more information on when to use this function see the
    /// documentation for post-return in the [`Func::call`] method.
    /// Otherwise though this function is a required method call after a
    /// [`Func::call`] completes successfully. After the embedder has
    /// finished processing the return value then this function must be invoked.
    ///
    /// # Errors
    ///
    /// This function will return an error in the case of a WebAssembly trap
    /// happening during the execution of the `post-return` function, if
    /// specified.
    ///
    /// # Panics
    ///
    /// This function will panic if it's not called under the correct
    /// conditions. This can only be called after a previous invocation of
    /// [`Func::call`] completes successfully, and this function can only
    /// be called for the same [`Func`] that was `call`'d.
    ///
    /// If this function is called when [`Func::call`] was not previously
    /// called, then it will panic. If a different [`Func`] for the same
    /// component instance was invoked then this function will also panic
    /// because the `post-return` needs to happen for the other function.
    pub fn post_return(&self, mut store: impl AsContextMut) -> Result<()> {
        let mut store = store.as_context_mut();
        let data = &mut store.0[self.0];
        let instance = data.instance;
        let post_return = data.post_return;
        let component_instance = data.component_instance;
        let post_return_arg = data.post_return_arg.take();
        let instance = store.0[instance.0].as_ref().unwrap().instance();
        let flags = instance.flags(component_instance);

        unsafe {
            // First assert that the instance is in a "needs post return" state.
            // This will ensure that the previous action on the instance was a
            // function call above. This flag is only set after a component
            // function returns so this also can't be called (as expected)
            // during a host import for example.
            //
            // Note, though, that this assert is not sufficient because it just
            // means some function on this instance needs its post-return
            // called. We need a precise post-return for a particular function
            // which is the second assert here (the `.expect`). That will assert
            // that this function itself needs to have its post-return called.
            //
            // The theory at least is that these two asserts ensure component
            // model semantics are upheld where the host properly calls
            // `post_return` on the right function despite the call being a
            // separate step in the API.
            assert!(
                (*flags).needs_post_return(),
                "post_return can only be called after a function has previously been called",
            );
            let post_return_arg = post_return_arg.expect("calling post_return on wrong function");

            // This is a sanity-check assert which shouldn't ever trip.
            assert!(!(*flags).may_enter());

            // Unset the "needs post return" flag now that post-return is being
            // processed. This will cause future invocations of this method to
            // panic, even if the function call below traps.
            (*flags).set_needs_post_return(false);

            // If the function actually had a `post-return` configured in its
            // canonical options that's executed here.
            //
            // Note that if this traps (returns an error) this function
            // intentionally leaves the instance in a "poisoned" state where it
            // can no longer be entered because `may_enter` is `false`.
            if let Some((func, trampoline)) = post_return {
                crate::Func::call_unchecked_raw(
                    &mut store,
                    func.anyfunc,
                    trampoline,
                    &post_return_arg as *const ValRaw as *mut ValRaw,
                )?;
            }

            // And finally if everything completed successfully then the "may
            // enter" flag is set to `true` again here which enables further use
            // of the component.
            (*flags).set_may_enter(true);
        }
        Ok(())
    }

    fn store_args<T>(
        &self,
        store: &mut StoreContextMut<'_, T>,
        options: &Options,
        params: &[Type],
        args: &[Val],
        dst: &mut MaybeUninit<[ValRaw; MAX_STACK_PARAMS]>,
    ) -> Result<()> {
        let mut size = 0;
        let mut alignment = 1;
        for ty in params {
            alignment = alignment.max(ty.size_and_alignment().alignment);
            ty.next_field(&mut size);
        }

        let mut memory = MemoryMut::new(store.as_context_mut(), options);
        let ptr = memory.realloc(0, 0, alignment, size)?;
        let mut offset = ptr;
        for (ty, arg) in params.iter().zip(args) {
            arg.store(&mut memory, ty.next_field(&mut offset))?;
        }

        map_maybe_uninit!(dst[0]).write(ValRaw::i64(ptr as i64));

        Ok(())
    }

    fn load_result<'a>(
        store: &StoreOpaque,
        mem: &Memory,
        ty: &Type,
        src: &mut impl Iterator<Item = &'a ValRaw>,
    ) -> Result<Val> {
        let SizeAndAlignment { size, alignment } = ty.size_and_alignment();
        // FIXME: needs to read an i64 for memory64
        let ptr = usize::try_from(src.next().unwrap().get_u32())?;
        if ptr % usize::try_from(alignment)? != 0 {
            bail!("return pointer not aligned");
        }

        let bytes = mem
            .as_slice()
            .get(ptr..)
            .and_then(|b| b.get(..size))
            .ok_or_else(|| anyhow::anyhow!("pointer out of bounds of memory"))?;

        ty.load(store, mem, bytes)
    }
}
