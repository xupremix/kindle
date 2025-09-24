use crate::{
    prelude::SqueezeDim,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    // Usually squeeze is used to remove all dimensions of size 1, here we remove only one dimension
    pub fn squeeze<const DIM: usize>(&self) -> Tensor<S::SqueezeShape, K, D>
    where
        S: SqueezeDim<DIM>,
    {
        Tensor {
            repr: self.repr.squeeze(DIM).unwrap(),
            ..Default::default()
        }
    }
}
