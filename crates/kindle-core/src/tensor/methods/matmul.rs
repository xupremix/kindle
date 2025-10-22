use crate::{
    prelude::Matmul,
    tensor::{Device, FromCandleTensor, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn matmul<Rhs: Matmul<S>>(
        &self,
        rhs: &Tensor<Rhs, K, D>,
    ) -> Tensor<Rhs::MatmulShape, K, D> {
        Tensor::from_candle_tensor(self.repr.matmul(&rhs.repr).unwrap())
    }
}
