use std::{fmt::Debug, marker::PhantomData};

use crate::{
    device::{Cpu, Device},
    kind::Kind,
    shape::Shape,
};

mod gen;

#[derive(Clone)]
pub struct Tensor<S: Shape, K: Kind = f32, D: Device = Cpu> {
    repr: candle_core::Tensor,
    __shape: PhantomData<S>,
    __kind: PhantomData<K>,
    __device: PhantomData<D>,
}

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub const fn dims() -> usize {
        S::DIMS
    }

    pub const fn nelems() -> usize {
        S::NELEMS
    }

    pub fn to_vec(&self) -> Vec<Vec<K>> {
        self.repr.to_vec2().unwrap()
    }
}

impl<S: Shape, K: Kind, D: Device> Default for Tensor<S, K, D> {
    fn default() -> Self {
        Self {
            repr: candle_core::Tensor::zeros(S::shape(), K::kind(), &D::device()).unwrap(),
            __shape: PhantomData,
            __kind: PhantomData,
            __device: PhantomData,
        }
    }
}

impl<S: Shape, K: Kind, D: Device> Debug for Tensor<S, K, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{:#?}", self.repr)
        } else {
            write!(f, "{:?}", self.repr)
        }
    }
}
