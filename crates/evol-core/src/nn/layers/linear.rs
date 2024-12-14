use std::{fmt::Debug, marker::PhantomData};

use candle_core::WithDType;
use candle_nn::Module as _;

use crate::{
    device::Device,
    nn::{Forward, Module},
    prelude::Vs,
    shape::{Rank1, Rank2},
    tensor::Tensor,
};

#[cfg(not(feature = "cuda"))]
use crate::device::Cpu;

#[cfg(feature = "cuda")]
use crate::device::Cuda;

#[cfg(feature = "cuda")]
pub struct Linear<
    const I: usize,
    const O: usize,
    const BIAS: bool = true,
    K: WithDType = f32,
    D: Device = Cuda,
> {
    repr: candle_nn::Linear,
    __kind: PhantomData<K>,
    __device: PhantomData<D>,
}

#[cfg(not(feature = "cuda"))]
pub struct Linear<
    const I: usize,
    const O: usize,
    const BIAS: bool = true,
    K: WithDType = f32,
    D: Device = Cpu,
> {
    repr: candle_nn::Linear,
    __kind: PhantomData<K>,
    __device: PhantomData<D>,
}

impl<const I: usize, const O: usize, const BIAS: bool, K: WithDType, D: Device>
    Linear<I, O, BIAS, K, D>
{
    pub const fn has_bias(&self) -> bool {
        BIAS
    }

    pub fn linear<S: ToString>(vs: &Vs<'_, K, D>, s: S) -> Self {
        Self {
            repr: candle_nn::linear_b(I, O, BIAS, vs.pp(s)).unwrap(),
            __kind: PhantomData,
            __device: PhantomData,
        }
    }

    pub fn weight(&self) -> Tensor<Rank2<O, I>, K, D> {
        Tensor {
            repr: self.repr.weight().clone(),
            ..Default::default()
        }
    }
}

impl<const I: usize, const O: usize, K: WithDType, D: Device> Linear<I, O, false, K, D> {
    pub fn new(w: Tensor<Rank2<O, I>, K, D>) -> Self {
        Self {
            repr: candle_nn::Linear::new(w.repr, None),
            __kind: PhantomData,
            __device: PhantomData,
        }
    }
}

impl<const I: usize, const O: usize, K: WithDType, D: Device> Linear<I, O, true, K, D> {
    pub fn new(w: Tensor<Rank2<O, I>, K, D>, b: Tensor<Rank1<O>, K, D>) -> Self {
        Self {
            repr: candle_nn::Linear::new(w.repr, Some(b.repr)),
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

impl<const I: usize, const O: usize, const BIAS: bool, K: WithDType, D: Device> Debug
    for Linear<I, O, BIAS, K, D>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{:#?}", self.repr)
        } else {
            write!(f, "{:?}", self.repr)
        }
    }
}

impl<const I: usize, const O: usize, const B: bool, S: Forward<I, O>, K: WithDType, D: Device>
    Module<Tensor<S, K, D>> for Linear<I, O, B, K, D>
{
    type Output = Tensor<S::ForwardShape, K, D>;

    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: self.repr.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}
