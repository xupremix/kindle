pub(crate) trait Sealed {}
pub use candle_core::op::CmpOp;

use crate::{
    device::Device,
    kind::Kind,
    shape::{Scalar, Shape},
    tensor::Tensor,
};

pub trait ToTensorScalar<S: Shape, K: Kind, D: Device>: Clone {
    fn to_tensorscalar(self) -> TensorScalar<S, K, D>;
}

pub enum TensorScalar<S: Shape, K: Kind, D: Device> {
    Tensor(Tensor<S, K, D>),
    Scalar(Tensor<Scalar, K, D>),
}

impl<S: Shape, K: Kind, D: Device> ToTensorScalar<S, K, D> for K {
    fn to_tensorscalar(self) -> TensorScalar<S, K, D> {
        TensorScalar::Scalar(Tensor::new(self))
    }
}
impl<S: Shape, K: Kind, D: Device> ToTensorScalar<S, K, D> for &K {
    fn to_tensorscalar(self) -> TensorScalar<S, K, D> {
        TensorScalar::Scalar(Tensor::new(self.clone()))
    }
}

impl<S: Shape, K: Kind, D: Device> ToTensorScalar<S, K, D> for Tensor<S, K, D> {
    fn to_tensorscalar(self) -> TensorScalar<S, K, D> {
        TensorScalar::Tensor(self)
    }
}
impl<S: Shape, K: Kind, D: Device> ToTensorScalar<S, K, D> for &Tensor<S, K, D> {
    fn to_tensorscalar(self) -> TensorScalar<S, K, D> {
        TensorScalar::Tensor(self.clone())
    }
}

pub trait ToUsize2 {
    const FIRST: usize;
    const SECOND: usize;
    const TUPLE: (usize, usize) = (Self::FIRST, Self::SECOND);
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Window<const H: usize, const W: usize = H>;
impl<const H: usize, const W: usize> ToUsize2 for Window<H, W> {
    const FIRST: usize = H;
    const SECOND: usize = W;
}

pub type Kernel<const H: usize, const W: usize = H> = Window<H, W>;
pub type Stride<const H: usize, const W: usize = H> = Window<H, W>;
