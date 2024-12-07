use crate::{
    kind::ToF64,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: candle_core::FloatDType, D: Device> Tensor<S, K, D> {
    pub fn random() -> Self {
        Self {
            repr: candle_core::Tensor::rand(K::zero(), K::one(), S::shape(), &D::device()).unwrap(),
            ..Default::default()
        }
    }

    pub fn random_like(&self) -> Self {
        Self::random()
    }

    pub fn rand(low: K, high: K) -> Self {
        Self {
            repr: candle_core::Tensor::rand(low, high, S::shape(), &D::device()).unwrap(),
            ..Default::default()
        }
    }

    pub fn rand_like<T: ToF64>(&self, low: T, high: T) -> Self {
        Self {
            repr: self.repr.rand_like(low.to_f64(), high.to_f64()).unwrap(),
            ..Default::default()
        }
    }
}
