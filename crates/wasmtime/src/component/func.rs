use crate::component::instance::{Instance, InstanceData};
use crate::component::values::Val;
use crate::store::{StoreOpaque, Stored};
use crate::{AsContext, AsContextMut, ValRaw};
use anyhow::{bail, Context, Result};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;
use wasmtime_environ::component::{
    CanonicalOptions, ComponentTypes, CoreDef, RuntimeComponentInstanceIndex, TypeFuncIndex,
};
use wasmtime_runtime::{Export, ExportFunction, VMTrampoline};

const MAX_STACK_PARAMS: usize = 16;
const MAX_STACK_RESULTS: usize = 1;

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

    pub fn call(&self, mut store: impl AsContextMut, params: &[Val]) -> Result<Val> {
        let store = &mut store.as_context_mut();

        let data = &store[self.0];
        let ty = &data.types[data.ty];

        if ty.params.len() != params.len() {
            bail!(
                "expected {} arguments, got {}",
                ty.params.len(),
                params.len()
            );
        }

        for ((_, ty), arg) in ty.params.iter().zip(params) {
            arg.typecheck(ty, &data.types)
                .context("type mismatch with parameters")?;
        }

        let FuncData {
            trampoline,
            export,
            options,
            instance,
            component_instance,
            ..
        } = store.0[self.0];

        let instance = store.0[instance.0].as_ref().unwrap().instance();
        let flags = instance.flags(component_instance);
        let mut space = Vec::new();

        // TODO: handle parameters passed via the heap

        unsafe {
            if !(*flags).may_enter() {
                bail!("cannot reenter component instance");
            }
            (*flags).set_may_enter(false);

            debug_assert!((*flags).may_leave());
            (*flags).set_may_leave(false);
            let result = params
                .iter()
                .map(|param| param.lower(store, &options, &mut space))
                .collect::<Result<()>>();
            (*flags).set_may_leave(true);
            result?;

            if space.is_empty() {
                // TODO: only do this if we're expecting a non-empty return value on the stack
                space.push(ValRaw::u32(0));
            }

            crate::Func::call_unchecked_raw(store, export.anyfunc, trampoline, space.as_mut_ptr())?;

            (*flags).set_needs_post_return(true);
        }

        // TODO: handle values returned via the heap

        let val = Val::lift(
            store.0,
            &options,
            &ty.result,
            &data.types,
            &mut space[..1].iter(),
        )?;
        let data = &mut store.0[self.0];
        assert!(data.post_return_arg.is_none());
        data.post_return_arg = Some(space[0]);
        Ok(val)
    }
}
