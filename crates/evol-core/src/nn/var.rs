use candle_core::WithDType;
use candle_nn::VarBuilder;
pub use candle_nn::VarMap;

use crate::device::Device;

#[cfg(not(feature = "cuda"))]
use crate::device::Cpu;

#[cfg(feature = "cuda")]
use crate::device::Cuda;

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct Vs<'a, K: WithDType = f32, D: Device = Cuda> {
    pub(crate) repr: VarBuilder<'a>,
    __kind: std::marker::PhantomData<K>,
    __device: std::marker::PhantomData<D>,
}

#[cfg(not(feature = "cuda"))]
#[derive(Clone)]
pub struct Vs<'a, K: WithDType = f32, D: Device = Cpu> {
    pub(crate) repr: VarBuilder<'a>,
    __kind: std::marker::PhantomData<K>,
    __device: std::marker::PhantomData<D>,
}

impl<'a, K: WithDType, D: Device> Vs<'a, K, D> {
    pub fn from_varmap(vm: &VarMap) -> Self {
        Self {
            repr: candle_nn::VarBuilder::from_varmap(vm, K::DTYPE, &D::device()),
            __kind: std::marker::PhantomData,
            __device: std::marker::PhantomData,
        }
    }

    /// alias for push_prefix
    #[inline(always)]
    pub fn pp<S: ToString>(&self, s: S) -> VarBuilder {
        self.repr.pp(s)
    }

    #[inline(always)]
    pub fn push_prefix<S: ToString>(&self, s: S) -> VarBuilder {
        self.pp(s)
    }

    #[inline(always)]
    pub fn root(&self) -> VarBuilder {
        self.repr.root()
    }

    #[inline(always)]
    pub fn prefix(&self) -> String {
        self.repr.prefix()
    }
}
