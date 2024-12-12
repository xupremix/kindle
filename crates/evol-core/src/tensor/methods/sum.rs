use crate::{
    prelude::Indexer,
    shape::Scalar,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn sum_all(&self) -> Tensor<Scalar, K, D> {
        Tensor {
            repr: self.repr.sum_all().unwrap(),
            ..Default::default()
        }
    }

    pub fn sum<I: Indexer<S, false>>(&self) -> Tensor<I::IndexShape, K, D> {
        Tensor {
            repr: self.repr.sum(I::indexes()).unwrap(),
            ..Default::default()
        }
    }

    pub fn sum_keepdim<I: Indexer<S, true>>(&self) -> Tensor<I::IndexShape, K, D> {
        Tensor {
            repr: self.repr.sum_keepdim(I::indexes()).unwrap(),
            ..Default::default()
        }
    }
}
