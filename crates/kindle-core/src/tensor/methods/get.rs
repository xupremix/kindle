use crate::{
    prelude::{Get, GetOnDim},
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

    pub fn get_on_dim<const DIM: usize, const N: usize>(&self) -> Tensor<S::GetOnDimShape, K, D>
    where
        S: GetOnDim<DIM, N>,
    {
        S::GET_ON_DIM_CHECK;
        Tensor {
            repr: self.repr.get_on_dim(DIM, N).unwrap(),
            ..Default::default()
        }
    }
}
