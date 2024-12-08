use crate::{
    prelude::Argmin,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn argmin<const DIM: usize>(&self) -> Tensor<S::ArgminShape, K, D>
    where
        S: Argmin<DIM, false>,
    {
        Tensor {
            repr: self.repr.argmin(DIM).unwrap(),
            ..Default::default()
        }
    }

    pub fn argmin_keepdim<const DIM: usize>(&self) -> Tensor<S::ArgminShape, K, D>
    where
        S: Argmin<DIM, true>,
    {
        Tensor {
            repr: self.repr.argmin_keepdim(DIM).unwrap(),
            ..Default::default()
        }
    }
}
