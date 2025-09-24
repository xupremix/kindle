use crate::{
    prelude::Matmul,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn matmul<Rhs: Matmul<S>>(
        &self,
        rhs: &Tensor<Rhs, K, D>,
    ) -> Tensor<Rhs::MatmulShape, K, D> {
        Tensor {
            repr: self.repr.matmul(&rhs.repr).unwrap(),
            ..Default::default()
        }
    }
}
