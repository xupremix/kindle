use std::{fmt::Debug, marker::PhantomData};

use candle_core::WithDType;
use candle_nn::{Conv2dConfig, Module as _};

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

#[cfg(feature = "cuda")]
pub struct Conv2d<
    const I: usize,
    const O: usize,
    const KERNEL: usize = 3,
    const PADDING: usize = 0,
    const STRIDE: usize = 1,
    const DILATION: usize = 1,
    const GROUPS: usize = 1,
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
    const PADDING: usize = 0,
    const STRIDE: usize = 1,
    const DILATION: usize = 1,
    const GROUPS: usize = 1,
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
        const PADDING: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const GROUPS: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > Conv2d<I, O, KERNEL, PADDING, STRIDE, DILATION, GROUPS, BIAS, K, D>
{
    pub const fn has_bias(&self) -> bool {
        BIAS
    }

    pub fn conv2d<S: ToString>(vs: &Vs<'_, K, D>, s: S) -> Self {
        let cfg = Conv2dConfig {
            padding: PADDING,
            stride: STRIDE,
            dilation: DILATION,
            groups: GROUPS,
        };
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

impl<
        const I: usize,
        const O: usize,
        const KERNEL: usize,
        const PADDING: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const GROUPS: usize,
        K: WithDType,
        D: Device,
    > Conv2d<I, O, KERNEL, PADDING, STRIDE, DILATION, GROUPS, true, K, D>
{
    pub fn new(
        weight: Tensor<Rank4<O, { I / GROUPS }, KERNEL, KERNEL>, K, D>,
        bias: Tensor<Rank1<O>, K, D>,
    ) -> Self {
        let cfg = Conv2dConfig {
            padding: PADDING,
            stride: STRIDE,
            dilation: DILATION,
            groups: GROUPS,
        };
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

impl<
        const I: usize,
        const O: usize,
        const KERNEL: usize,
        const PADDING: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const GROUPS: usize,
        K: WithDType,
        D: Device,
    > Conv2d<I, O, KERNEL, PADDING, STRIDE, DILATION, GROUPS, false, K, D>
{
    pub fn new(weight: Tensor<Rank4<O, { I / GROUPS }, KERNEL, KERNEL>, K, D>) -> Self {
        let cfg = Conv2dConfig {
            padding: PADDING,
            stride: STRIDE,
            dilation: DILATION,
            groups: GROUPS,
        };
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
        const PADDING: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const GROUPS: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > Debug for Conv2d<I, O, KERNEL, PADDING, STRIDE, DILATION, GROUPS, BIAS, K, D>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{:#?}", self.repr)
        } else {
            write!(f, "{:?}", self.repr)
        }
    }
}

///
/// Input: (N, Cin, Hin, Win)
/// Output: (N, Cout, Hout, Wout)
///
/// Hout = floor{ ( Hin + 2 * PADDING - DILATION * ( KERNEL - 1 ) ) / stride + 1 }
/// Wout = floor{ ( Win + 2 * PADDING - DILATION * ( KERNEL - 1 ) ) / stride + 1 }
///
impl<
        const N: usize,
        const CIN: usize,
        const HIN: usize,
        const WIN: usize,
        const O: usize,
        const KERNEL: usize,
        const PADDING: usize,
        const STRIDE: usize,
        const DILATION: usize,
        const GROUPS: usize,
        const BIAS: bool,
        K: WithDType,
        D: Device,
    > Module<Tensor<Rank4<N, CIN, HIN, WIN>, K, D>>
    for Conv2d<CIN, O, KERNEL, PADDING, STRIDE, DILATION, GROUPS, BIAS, K, D>
where
    [(); STRIDE]:,
    [(); CIN / GROUPS]:,
    [(); (CIN % GROUPS == 0) as usize]:, // TODO: CHECK if it's necessary
    [(); (O % GROUPS == 0) as usize]:,   // TODO: CHECK if it's necessary
    [(); (HIN + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1]:,
    [(); (WIN + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1]:,
{
    type Output = Tensor<
        Rank4<
            N,
            O,
            { (HIN + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 },
            { (WIN + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 },
        >,
        K,
        D,
    >;

    fn forward(&self, xs: &Tensor<Rank4<N, CIN, HIN, WIN>, K, D>) -> Self::Output {
        Tensor {
            repr: self.repr.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}
