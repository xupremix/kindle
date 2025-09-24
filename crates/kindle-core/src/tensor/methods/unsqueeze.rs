use crate::{
    prelude::Unsqueeze,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn unsqueeze<const DIM: usize>(&self) -> Tensor<S::UnsqueezeShape, K, D>
    where
        S: Unsqueeze<DIM>,
    {
        Tensor {
            repr: self.repr.unsqueeze(DIM).unwrap(),
            ..Default::default()
        }
    }
}
