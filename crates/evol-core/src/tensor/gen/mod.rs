use crate::device::Device;
use crate::kind::Kind;
use crate::tensor::Shape;
use crate::tensor::Tensor;

use std::marker::PhantomData;

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn ones() -> Self {
        Self {
            repr: candle_core::Tensor::ones(S::shape(), K::kind(), &D::device()).unwrap(),
            ..Default::default()
        }
    }

    pub fn ones_like(&self) -> Self {
        Self {
            repr: self.repr.ones_like().unwrap(),
            ..Default::default()
        }
    }

    pub fn zeros() -> Self {
        Default::default()
    }

    pub fn zeros_like(&self) -> Self {
        Default::default()
    }

    pub fn from_slice(arr: &S::Shape<K>) -> Self {
        Self {
            repr: candle_core::Tensor::from_slice(S::as_slice(arr), S::shape(), &D::device())
                .unwrap(),
            ..Default::default()
        }
    }

    pub fn new(arr: S::Shape<K>) -> Self {
        Self {
            repr: candle_core::Tensor::from_slice(S::as_slice(&arr), S::shape(), &D::device())
                .unwrap(),
            ..Default::default()
        }
    }
}

/*
    from_slice
    from_vec
    full
    narrow
    new
    ones
    ones_like
    rand
    rand_like
    randn
    randn_like
    zeros
    zeros_like
    arange
    arange_step
*/
