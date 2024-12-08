use crate::{
    prelude::Argmax,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn argmax<const DIM: usize>(&self) -> Tensor<S::ArgmaxShape, K, D>
    where
        S: Argmax<DIM, false>,
    {
        Tensor {
            repr: self.repr.argmax(DIM).unwrap(),
            ..Default::default()
        }
    }

    pub fn argmax_keepdim<const DIM: usize>(&self) -> Tensor<S::ArgmaxShape, K, D>
    where
        S: Argmax<DIM, true>,
    {
        Tensor {
            repr: self.repr.argmax_keepdim(DIM).unwrap(),
            ..Default::default()
        }
    }
}
