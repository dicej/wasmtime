#[cfg(feature = "component-model")]
pub use component::*;

#[cfg(feature = "component-model")]
mod component {
    use anyhow::Result;
    use arbitrary::Arbitrary;
    use std::mem::MaybeUninit;
    use wasmtime::component::__internal::{
        ComponentTypes, InterfaceType, Memory, MemoryMut, Options, StoreOpaque,
    };
    use wasmtime::component::{ComponentParams, ComponentType, Func, Lift, Lower, TypedFunc, Val};
    use wasmtime::{AsContextMut, Config, Engine, StoreContextMut};

    pub trait TypedFuncExt<P, R> {
        fn call_and_post_return(&self, store: impl AsContextMut, params: P) -> Result<R>;
    }

    impl<P, R> TypedFuncExt<P, R> for TypedFunc<P, R>
    where
        P: ComponentParams + Lower,
        R: Lift,
    {
        fn call_and_post_return(&self, mut store: impl AsContextMut, params: P) -> Result<R> {
            let result = self.call(&mut store, params)?;
            self.post_return(&mut store)?;
            Ok(result)
        }
    }

    pub trait FuncExt {
        fn call_and_post_return(&self, store: impl AsContextMut, args: &[Val]) -> Result<Val>;
    }

    impl FuncExt for Func {
        fn call_and_post_return(&self, mut store: impl AsContextMut, args: &[Val]) -> Result<Val> {
            let result = self.call(&mut store, args)?;
            self.post_return(&mut store)?;
            Ok(result)
        }
    }

    pub fn engine() -> Engine {
        let mut config = Config::new();
        config.wasm_component_model(true);
        Engine::new(&config).unwrap()
    }

    /// A simple bump allocator which can be used with modules
    pub const REALLOC_AND_FREE: &str = r#"
        (global $last (mut i32) (i32.const 8))
        (func $realloc (export "realloc")
            (param $old_ptr i32)
            (param $old_size i32)
            (param $align i32)
            (param $new_size i32)
            (result i32)

            ;; Test if the old pointer is non-null
            local.get $old_ptr
            if
                ;; If the old size is bigger than the new size then
                ;; this is a shrink and transparently allow it
                local.get $old_size
                local.get $new_size
                i32.gt_u
                if
                    local.get $old_ptr
                    return
                end

                ;; ... otherwise this is unimplemented
                unreachable
            end

            ;; align up `$last`
            (global.set $last
                (i32.and
                    (i32.add
                        (global.get $last)
                        (i32.add
                            (local.get $align)
                            (i32.const -1)))
                    (i32.xor
                        (i32.add
                            (local.get $align)
                            (i32.const -1))
                        (i32.const -1))))

            ;; save the current value of `$last` as the return value
            global.get $last

            ;; ensure anything necessary is set to valid data by spraying a bit
            ;; pattern that is invalid
            global.get $last
            i32.const 0xde
            local.get $new_size
            memory.fill

            ;; bump our pointer
            (global.set $last
                (i32.add
                    (global.get $last)
                    (local.get $new_size)))
        )
    "#;

    /// Newtype wrapper for `f32` whose `PartialEq` impl considers NaNs equal to each other.
    #[derive(Copy, Clone, Debug, Arbitrary)]
    pub struct Float32(pub f32);

    /// Newtype wrapper for `f64` whose `PartialEq` impl considers NaNs equal to each other.
    #[derive(Copy, Clone, Debug, Arbitrary)]
    pub struct Float64(pub f64);

    macro_rules! forward_impls {
        ($($a:ty => $b:ty,)*) => ($(
            unsafe impl ComponentType for $a {
                type Lower = <$b as ComponentType>::Lower;

                const SIZE32: usize = <$b as ComponentType>::SIZE32;
                const ALIGN32: u32 = <$b as ComponentType>::ALIGN32;

                #[inline]
                fn typecheck(ty: &InterfaceType, types: &ComponentTypes) -> Result<()> {
                    <$b as ComponentType>::typecheck(ty, types)
                }
            }

            unsafe impl Lower for $a {
                fn lower<U>(
                    &self,
                    store: &mut StoreContextMut<U>,
                    options: &Options,
                    dst: &mut MaybeUninit<Self::Lower>,
                ) -> Result<()> {
                    <$b as Lower>::lower(&self.0, store, options, dst)
                }

                fn store<U>(&self, memory: &mut MemoryMut<'_, U>, offset: usize) -> Result<()> {
                    <$b as Lower>::store(&self.0, memory, offset)
                }
            }

            unsafe impl Lift for $a {
                fn lift(store: &StoreOpaque, options: &Options, src: &Self::Lower) -> Result<Self> {
                    Ok(Self(<$b as Lift>::lift(store, options, src)?))
                }

                fn load(memory: &Memory<'_>, bytes: &[u8]) -> Result<Self> {
                    Ok(Self(<$b as Lift>::load(memory, bytes)?))
                }
            }

            impl PartialEq for $a {
                fn eq(&self, other: &Self) -> bool {
                    self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
                }
            }
        )*)
    }

    forward_impls! {
        Float32 => f32,
        Float64 => f64,
    }
}
