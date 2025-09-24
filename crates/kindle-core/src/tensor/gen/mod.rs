use crate::prelude::FromSlice;
use crate::tensor::{Device, Kind, Shape, Tensor};

mod arange;
mod rand;

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn ones() -> Self {
        Self {
            repr: candle_core::Tensor::ones(S::shape(), K::DTYPE, &D::device()).unwrap(),
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

    pub fn full(value: K) -> Self {
        Self {
            repr: candle_core::Tensor::full(value, S::shape(), &D::device()).unwrap(),
            ..Default::default()
        }
    }

    // consider using *const K if performance is necessary
    pub fn from_arr(arr: &S::Shape<K>) -> Self {
        Self {
            repr: candle_core::Tensor::from_slice(S::as_slice(arr), S::shape(), &D::device())
                .unwrap(),
            ..Default::default()
        }
    }

    // consider using *const K if performance is necessary
    pub fn new(arr: S::Shape<K>) -> Self {
        Self {
            repr: candle_core::Tensor::from_slice(S::as_slice(&arr), S::shape(), &D::device())
                .unwrap(),
            ..Default::default()
        }
    }
}

impl<'a, const N: usize, S: Shape, K: Kind, D: Device> FromSlice<&'a [K; N]> for Tensor<S, K, D> {
    const FROM_SLICE_CHECK: () = assert!(
        N == S::NELEMS,
        "The slice must have the same number of elements"
    );
    fn from_slice(slice: &'a [K; N]) -> Self {
        <Self as FromSlice<&[K; N]>>::FROM_SLICE_CHECK;
        Self {
            repr: candle_core::Tensor::from_slice(slice, S::shape(), &D::device()).unwrap(),
            ..Default::default()
        }
    }
}

impl<const N: usize, S: Shape, K: Kind, D: Device> FromSlice<[K; N]> for Tensor<S, K, D> {
    const FROM_SLICE_CHECK: () = assert!(
        N == S::NELEMS,
        "The slice must have the same number of elements"
    );
    fn from_slice(slice: [K; N]) -> Self {
        <Self as FromSlice<&[K; N]>>::FROM_SLICE_CHECK;
        Self {
            repr: candle_core::Tensor::from_slice(&slice, S::shape(), &D::device()).unwrap(),
            ..Default::default()
        }
    }
}
