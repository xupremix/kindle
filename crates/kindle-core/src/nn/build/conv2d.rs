use std::sync::atomic::Ordering;

use candle_core::WithDType;
use candle_nn::Conv2dConfig as Cfg;

use crate::{
    device::Device,
    prelude::{Conv2d, PREFIX},
    utils::ToUsize2,
};

use super::ModelBuilder;

pub struct Conv2dConfig(String);
impl Conv2dConfig {
    pub fn new<S: ToString>(s: S) -> Self {
        Self(s.to_string())
    }
}
impl Default for Conv2dConfig {
    fn default() -> Self {
        Self(format!("conv2d_{}", PREFIX.fetch_add(1, Ordering::Relaxed)))
    }
}

impl<
        const I: usize,
        const O: usize,
        Kernel: ToUsize2,
        const PADDING: usize,
        Stride: ToUsize2,
        const DILATION: usize,
        const GROUPS: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > ModelBuilder for Conv2d<I, O, Kernel, PADDING, Stride, DILATION, GROUPS, BIAS, K, D>
{
    type Config = Conv2dConfig;

    fn step(
        vs: &crate::prelude::Vs,
        cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        let cfg2d = Cfg {
            padding: PADDING,
            stride: Stride::FIRST,
            dilation: DILATION,
            groups: GROUPS,
            cudnn_fwd_algo: None,
        };
        seq.add(if BIAS {
            candle_nn::conv2d(I, O, Kernel::FIRST, cfg2d, vs.pp(cfg.0)).unwrap()
        } else {
            candle_nn::conv2d_no_bias(I, O, Kernel::FIRST, cfg2d, vs.pp(cfg.0)).unwrap()
        })
    }
}
