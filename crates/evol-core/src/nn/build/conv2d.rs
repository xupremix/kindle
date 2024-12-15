use std::sync::atomic::Ordering;

use candle_core::WithDType;
use candle_nn::Conv2dConfig;

use crate::{
    device::Device,
    prelude::{Conv2d, PREFIX},
};

use super::ModelBuilder;

pub struct Conv2dBuildConfig(Conv2dConfig, String);
impl Conv2dBuildConfig {
    pub fn new<S: ToString>(cfg: Conv2dConfig, s: S) -> Self {
        Self(cfg, s.to_string())
    }
}
impl Default for Conv2dBuildConfig {
    fn default() -> Self {
        Self(
            Default::default(),
            format!("linear_{}", PREFIX.fetch_add(1, Ordering::Relaxed)),
        )
    }
}

impl<
        const I: usize,
        const O: usize,
        const KERNEL: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > ModelBuilder for Conv2d<I, O, KERNEL, BIAS, K, D>
{
    type Config = Conv2dBuildConfig;

    fn step(
        vs: &crate::prelude::Vs,
        cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add(if BIAS {
            candle_nn::conv2d(I, O, KERNEL, cfg.0, vs.pp(cfg.1)).unwrap()
        } else {
            candle_nn::conv2d_no_bias(I, O, KERNEL, cfg.0, vs.pp(cfg.1)).unwrap()
        })
    }
}
