use crate::{
    prelude::Reshape,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn reshape<S2: Reshape<S>>(&self) -> Tensor<S2, K, D> {
        S2::RESHAPE_CHECK;
        Tensor {
            repr: self.repr.reshape(S2::shape()).unwrap(),
            ..Default::default()
        }
    }

    pub fn reshape_like<S2: Shape>(&self, _: &Tensor<S2, K, D>) -> Tensor<S2, K, D> {
        self.reshape()
    }
}
