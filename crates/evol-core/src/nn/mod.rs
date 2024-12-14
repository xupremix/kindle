use candle_core::WithDType;

use crate::{
    device::Device,
    shape::Shape,
    tensor::{FromCandleTensor, Tensor},
};

pub mod build;
pub mod layers;
pub mod optim;
pub mod vs;

pub mod prelude {
    pub use super::build::*;
    pub use super::layers::conv2d::*;
    pub use super::layers::linear::*;
    pub use super::optim::adam::*;
    pub use super::optim::sgd::*;
    pub use super::vs::*;
}

pub trait Module<T> {
    type Output: FromCandleTensor;
    fn forward(&self, xs: &T) -> Self::Output;
}

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}

impl<S, K, D, M0, M1, M2> Module<Tensor<S, K, D>> for (M0, M1, M2)
where
    S: Shape,
    K: WithDType,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
{
    type Output = M2::Output;

    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        self.2.forward(&xs)
    }
}
