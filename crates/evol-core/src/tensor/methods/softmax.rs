use crate::{
    prelude::DimInRange,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn softmax<const DIM: usize>(&self) -> Self
    where
        S: DimInRange<DIM>,
    {
        S::DIM_IN_RANGE_CHECK;
        Self {
            repr: candle_nn::ops::softmax(&self.repr, DIM).unwrap(),
            ..Default::default()
        }
    }

    pub fn log_softmax<const DIM: usize>(&self) -> Self
    where
        S: DimInRange<DIM>,
    {
        S::DIM_IN_RANGE_CHECK;
        Self {
            repr: candle_nn::ops::log_softmax(&self.repr, DIM).unwrap(),
            ..Default::default()
        }
    }

    pub fn softmax_last_dim(&self) -> Self {
        Self {
            repr: candle_nn::ops::softmax_last_dim(&self.repr).unwrap(),
            ..Default::default()
        }
    }
}
