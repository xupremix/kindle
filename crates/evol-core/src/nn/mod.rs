use crate::{shape::Shape, tensor::FromCandleTensor};

pub mod build;
pub mod layers;
pub mod optim;
mod tuple;
pub mod vs;

pub mod prelude {
    pub use super::build::*;
    pub use super::layers::activations::*;
    #[cfg(feature = "nightly")]
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
