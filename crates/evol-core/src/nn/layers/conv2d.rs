use std::{fmt::Debug, marker::PhantomData};

use candle_core::WithDType;
use candle_nn::Module as _;

use crate::{
    device::Device,
    nn::Module,
    prelude::Vs,
    shape::{Rank1, Rank4},
    tensor::Tensor,
};

#[cfg(not(feature = "cuda"))]
use crate::device::Cpu;

#[cfg(feature = "cuda")]
use crate::device::Cuda;

pub use candle_nn::Conv2dConfig;

#[cfg(feature = "cuda")]
pub struct Conv2d<
    const I: usize,
    const O: usize,
    const KERNEL: usize = 3,
    const BIAS: bool = true,
    K: WithDType = f32,
    D: Device = Cuda,
> {
    repr: candle_nn::Conv2d,
    __kind: PhantomData<K>,
    __device: PhantomData<D>,
}

#[cfg(not(feature = "cuda"))]
pub struct Conv2d<
    const I: usize,
    const O: usize,
    const KERNEL: usize = 3,
    const BIAS: bool = true,
    K: WithDType = f32,
    D: Device = Cpu,
> {
    repr: candle_nn::Conv2d,
    __kind: PhantomData<K>,
    __device: PhantomData<D>,
}

impl<
        const I: usize,
        const O: usize,
        const KERNEL: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > Conv2d<I, O, KERNEL, BIAS, K, D>
{
    pub const fn has_bias(&self) -> bool {
        BIAS
    }

    pub fn conv2d<S: ToString>(vs: &Vs<'_, K, D>, s: S, cfg: Conv2dConfig) -> Self {
        Self {
            repr: if BIAS {
                candle_nn::conv2d(I, O, KERNEL, cfg, vs.pp(s)).unwrap()
            } else {
                candle_nn::conv2d_no_bias(I, O, KERNEL, cfg, vs.pp(s)).unwrap()
            },
            __kind: PhantomData,
            __device: PhantomData,
        }
    }

    pub fn config(&self) -> &Conv2dConfig {
        self.repr.config()
    }

    pub fn weight(&self) -> Tensor<Rank4<O, I, KERNEL, KERNEL>, K, D> {
        Tensor {
            repr: self.repr.weight().clone(),
            ..Default::default()
        }
    }
}

impl<const I: usize, const O: usize, const KERNEL: usize, K: WithDType, D: Device>
    Conv2d<I, O, KERNEL, true, K, D>
{
    pub fn new(
        weight: Tensor<Rank4<O, I, KERNEL, KERNEL>, K, D>,
        bias: Tensor<Rank1<O>, K, D>,
        cfg: Conv2dConfig,
    ) -> Self {
        Self {
            repr: candle_nn::Conv2d::new(weight.repr, Some(bias.repr), cfg),
            __kind: PhantomData,
            __device: PhantomData,
        }
    }

    pub fn bias(&self) -> Tensor<Rank1<O>, K, D> {
        Tensor {
            repr: self.repr.bias().unwrap().clone(),
            ..Default::default()
        }
    }
}

impl<const I: usize, const O: usize, const KERNEL: usize, K: WithDType, D: Device>
    Conv2d<I, O, KERNEL, false, K, D>
{
    pub fn new(weight: Tensor<Rank4<O, I, KERNEL, KERNEL>, K, D>, cfg: Conv2dConfig) -> Self {
        Self {
            repr: candle_nn::Conv2d::new(weight.repr, None, cfg),
            __kind: PhantomData,
            __device: PhantomData,
        }
    }
}

impl<
        const I: usize,
        const O: usize,
        const KERNEL: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > Debug for Conv2d<I, O, KERNEL, BIAS, K, D>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{:#?}", self.repr)
        } else {
            write!(f, "{:?}", self.repr)
        }
    }
}

impl<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const O: usize,
        const KERNEL: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > Module<Tensor<Rank4<D0, D1, D2, D3>, K, D>> for Conv2d<D1, O, KERNEL, BIAS, K, D>
where
    [(); D2 + 1 - KERNEL]: Sized,
    [(); D3 + 1 - KERNEL]: Sized,
{
    type Output = Tensor<Rank4<D0, O, { D2 + 1 - KERNEL }, { D3 + 1 - KERNEL }>, K, D>;

    fn forward(&self, xs: &Tensor<Rank4<D0, D1, D2, D3>, K, D>) -> Self::Output {
        Tensor {
            repr: self.repr.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}
