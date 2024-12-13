use crate::{
    prelude::Get,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn get<const N: usize>(&self) -> Tensor<S::GetShape, K, D>
    where
        S: Get<N>,
    {
        S::GET_CHECK;
        Tensor {
            repr: self.repr.get(N).unwrap(),
            ..Default::default()
        }
    }
}
