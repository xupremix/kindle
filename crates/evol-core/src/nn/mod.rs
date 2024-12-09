use crate::shape::Shape;

pub mod layers;
pub mod var;

pub mod prelude {
    pub use super::layers::prelude::*;
    pub use super::layers::*;
    pub use super::var::*;
}

pub trait Module<T> {
    type Output;
    fn forward(&self, xs: T) -> Self::Output;
}

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}
