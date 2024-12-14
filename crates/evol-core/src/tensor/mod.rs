use safetensors::tensor::View;
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
    marker::PhantomData,
};

use crate::{device::Device, kind::Kind, shape::Shape};

#[cfg(feature = "cuda")]
use crate::device::Cuda;

#[cfg(not(feature = "cuda"))]
use crate::device::Cpu;

mod broadcast;
mod cmp;
mod gen;
mod math;
mod methods;
mod ops;
mod wrap;

pub mod method_traits;

pub mod prelude {
    pub use super::method_traits::*;
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct Tensor<S: Shape, K: Kind = f32, D: Device = Cuda> {
    pub(crate) repr: candle_core::Tensor,
    pub(crate) __shape: PhantomData<S>,
    pub(crate) __kind: PhantomData<K>,
    pub(crate) __device: PhantomData<D>,
}

#[cfg(not(feature = "cuda"))]
#[derive(Clone)]
pub struct Tensor<S: Shape, K: Kind = f32, D: Device = Cpu> {
    pub(crate) repr: candle_core::Tensor,
    pub(crate) __shape: PhantomData<S>,
    pub(crate) __kind: PhantomData<K>,
    pub(crate) __device: PhantomData<D>,
}

pub(crate) trait FromCandleTensor {
    fn from_candle_tensor(repr: candle_core::Tensor) -> Self;
}

impl<S: Shape, K: Kind, D: Device> FromCandleTensor for Tensor<S, K, D> {
    fn from_candle_tensor(repr: candle_core::Tensor) -> Self {
        Self {
            repr,
            ..Default::default()
        }
    }
}

pub trait ToCandleTensor {
    fn to_candle_tensor(&self) -> &candle_core::Tensor;
}

impl<S: Shape, K: Kind, D: Device> ToCandleTensor for Tensor<S, K, D> {
    fn to_candle_tensor(&self) -> &candle_core::Tensor {
        &self.repr
    }
}

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub const fn shape_rank() -> usize {
        S::DIMS
    }

    pub const fn rank(&self) -> usize {
        S::DIMS
    }

    pub const fn shape_nelems() -> usize {
        S::NELEMS
    }

    pub const fn nelems() -> usize {
        S::NELEMS
    }

    pub fn shape_shape() -> &'static [usize] {
        S::dims()
    }

    pub fn shape(&self) -> &'static [usize] {
        S::dims()
    }

    pub fn stride(&self) -> &[usize] {
        self.repr.stride()
    }

    pub fn data(&self) -> Cow<'_, [u8]> {
        self.repr.data()
    }

    pub fn data_len(&self) -> usize {
        self.repr.data_len()
    }

    pub fn to_kind<K2: Kind>(&self) -> Tensor<S, K2, D> {
        Tensor {
            repr: self.repr.to_dtype(K2::DTYPE).unwrap(),
            __shape: PhantomData,
            __kind: PhantomData,
            __device: PhantomData,
        }
    }

    pub fn to_kind_like<K2: Kind>(&self, _: &Tensor<S, K2, D>) -> Tensor<S, K2, D> {
        Tensor {
            repr: self.repr.to_dtype(K2::DTYPE).unwrap(),
            __shape: PhantomData,
            __kind: PhantomData,
            __device: PhantomData,
        }
    }

    pub fn to_device<D2: Device>(&self) -> Tensor<S, K, D2> {
        Tensor {
            repr: self.repr.to_device(&D2::device()).unwrap(),
            __shape: PhantomData,
            __kind: PhantomData,
            __device: PhantomData,
        }
    }

    pub fn to_device_like<D2: Device>(&self, _: &Tensor<S, K, D2>) -> Tensor<S, K, D2> {
        Tensor {
            repr: self.repr.to_device(&D2::device()).unwrap(),
            __shape: PhantomData,
            __kind: PhantomData,
            __device: PhantomData,
        }
    }
}

impl<S: Shape, K: Kind, D: Device> Default for Tensor<S, K, D> {
    fn default() -> Self {
        Self {
            repr: candle_core::Tensor::zeros(S::shape(), K::DTYPE, &D::device()).unwrap(),
            __shape: PhantomData,
            __kind: PhantomData,
            __device: PhantomData,
        }
    }
}

impl<S: Shape, K: Kind, D: Device> Debug for Tensor<S, K, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{:#?}", self.repr)
        } else {
            write!(f, "{:?}", self.repr)
        }
    }
}

impl<S: Shape, K: Kind, D: Device> Display for Tensor<S, K, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{:#}", self.repr)
        } else {
            write!(f, "{}", self.repr)
        }
    }
}
