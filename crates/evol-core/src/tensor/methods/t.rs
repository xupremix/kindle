use crate::{
    prelude::T,
    tensor::{Device, Kind, Tensor},
};

impl<S: T, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn t(&self) -> Tensor<S::Transposed, K, D> {
        Tensor {
            repr: self.repr.t().unwrap(),
            ..Default::default()
        }
    }
}
