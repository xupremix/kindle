use crate::{
    prelude::{DimInRange, SwigluShape},
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn swiglu(&self) -> Tensor<S::SwigluShape, K, D>
    where
        S: SwigluShape,
    {
        Tensor {
            repr: candle_nn::ops::swiglu(&self.repr).unwrap(),
            ..Default::default()
        }
    }

    pub fn sigmoid(&self) -> Self {
        Self {
            repr: candle_nn::ops::sigmoid(&self.repr).unwrap(),
            ..Default::default()
        }
    }

    pub fn dropout(&self, p: f32) -> Self {
        Self {
            repr: candle_nn::ops::dropout(&self.repr, p).unwrap(),
            ..Default::default()
        }
    }

    pub fn hard_sigmoid(&self) -> Self {
        Self {
            repr: candle_nn::ops::hard_sigmoid(&self.repr).unwrap(),
            ..Default::default()
        }
    }

    pub fn leaky_relu(&self, negative_slope: f64) -> Self {
        Self {
            repr: candle_nn::ops::leaky_relu(&self.repr, negative_slope).unwrap(),
            ..Default::default()
        }
    }

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
