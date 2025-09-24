use std::sync::atomic::Ordering;

use candle_core::WithDType;

use crate::{
    device::Device,
    prelude::{Linear, PREFIX},
};

use super::ModelBuilder;

pub struct LinearConfig(String);
impl LinearConfig {
    pub fn new<S: ToString>(s: S) -> Self {
        Self(s.to_string())
    }
}
impl Default for LinearConfig {
    fn default() -> Self {
        Self(format!("linear_{}", PREFIX.fetch_add(1, Ordering::Relaxed)))
    }
}

impl<const I: usize, const O: usize, const BIAS: bool, K: WithDType, D: Device> ModelBuilder
    for Linear<I, O, BIAS, K, D>
{
    type Config = LinearConfig;

    fn step(
        vs: &crate::prelude::Vs,
        cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add(candle_nn::linear_b(I, O, BIAS, vs.pp(cfg.0)).unwrap())
    }
}
