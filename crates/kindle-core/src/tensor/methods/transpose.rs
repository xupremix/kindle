use crate::{
    prelude::Transpose,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn transpose<const D0: usize, const D1: usize>(&self) -> Tensor<S::Transposed, K, D>
    where
        S: Transpose<D0, D1>,
    {
        Tensor {
            repr: self.repr.transpose(D0, D1).unwrap(),
            ..Default::default()
        }
    }
}
