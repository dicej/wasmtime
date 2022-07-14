use crate::component::instance::{Instance, InstanceData};
use crate::component::values::{self, SizeAndAlignment, Type, Val};
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
                bail!("expected {} arguments, got {}", ty.params.len(), args.len());
            }

            params = ty
                .params
                .iter()
                .zip(args)
                .map(|((_, ty), arg)| {
                    let ty = Type::from(ty, &data.types);

                    arg.typecheck(&ty)
                        .context("type mismatch with parameters")?;

                    Ok(ty)
                })
                .collect::<Result<Vec<_>>>()?;

            result = Type::from(&ty.result, &data.types);
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
        let param_count = params.iter().map(|ty| ty.flatten_count()).sum::<usize>();
        let result_count = result.flatten_count();
        let mut space = Vec::new();

        unsafe {
            if !(*flags).may_enter() {
                bail!("cannot reenter component instance");
            }
            (*flags).set_may_enter(false);

            debug_assert!((*flags).may_leave());
            (*flags).set_may_leave(false);
            let result = if param_count > MAX_STACK_PARAMS {
                self.store_args(store, &options, &params, args, &mut space)
            } else {
                params
                    .iter()
                    .zip(args)
                    .try_for_each(|(ty, arg)| arg.lower(store, &options, ty, &mut space))
            };
            (*flags).set_may_leave(true);
            result?;

            if space.is_empty() && result_count > 0 {
                // reserve space for return value
                space.push(ValRaw::u32(0));
            }

            crate::Func::call_unchecked_raw(store, export.anyfunc, trampoline, space.as_mut_ptr())?;

            (*flags).set_needs_post_return(true);
        }

        let val = if result_count > MAX_STACK_RESULTS {
            Self::load_result(
                store.0,
                &Memory::new(store.0, &options),
                &result,
                &mut space[..MAX_STACK_RESULTS].iter(),
            )
        } else {
            Val::lift(
                store.0,
                &options,
                &result,
                &mut space[..MAX_STACK_RESULTS].iter(),
            )
        }?;

        let data = &mut store.0[self.0];
        assert!(data.post_return_arg.is_none());
        data.post_return_arg = Some(space[0]);
        Ok(val)
    }

    fn store_args<T>(
        &self,
        store: &mut StoreContextMut<'_, T>,
        options: &Options,
        params: &[Type],
        args: &[Val],
        vec: &mut Vec<ValRaw>,
    ) -> Result<()> {
        let mut size = 0;
        let mut alignment = 1;
        for ty in params {
            alignment = alignment.max(SizeAndAlignment::from(ty).alignment);
            values::next_field(ty, &mut size);
        }

        let mut memory = MemoryMut::new(store.as_context_mut(), options);
        let ptr = memory.realloc(0, 0, alignment, size)?;
        let mut offset = ptr;
        for (ty, arg) in params.iter().zip(args) {
            arg.store(&mut memory, ty, values::next_field(ty, &mut offset))?;
        }

        vec.push(ValRaw::i64(ptr as i64));

        Ok(())
    }

    fn load_result<'a>(
        store: &StoreOpaque,
        mem: &Memory,
        ty: &Type,
        src: &mut impl Iterator<Item = &'a ValRaw>,
    ) -> Result<Val> {
        let SizeAndAlignment { size, alignment } = SizeAndAlignment::from(ty);
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

        Val::load(store, mem, ty, bytes)
    }
}
