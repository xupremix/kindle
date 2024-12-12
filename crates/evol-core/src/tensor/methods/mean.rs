use crate::{
    prelude::Indexer,
    shape::Scalar,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn mean_all(&self) -> Tensor<Scalar, K, D> {
        Tensor {
            repr: self.repr.mean_all().unwrap(),
            ..Default::default()
        }
    }

    pub fn mean<I: Indexer<S, false>>(&self) -> Tensor<I::IndexShape, K, D> {
        Tensor {
            repr: self.repr.mean(I::indexes()).unwrap(),
            ..Default::default()
        }
    }

    pub fn mean_keepdim<I: Indexer<S, true>>(&self) -> Tensor<I::IndexShape, K, D> {
        Tensor {
            repr: self.repr.mean_keepdim(I::indexes()).unwrap(),
            ..Default::default()
        }
    }
}
