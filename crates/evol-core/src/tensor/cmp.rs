use candle_core::{op::CmpOp, InplaceOp1};
use safetensors::View;

use crate::tensor::{Device, Kind, Shape, Tensor};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn lt(&self, other: &Self) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.lt(&other.repr).unwrap(),
            ..Default::default()
        }
    }
    pub fn le(&self, other: &Self) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.le(&other.repr).unwrap(),
            ..Default::default()
        }
    }
    pub fn gt(&self, other: &Self) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.gt(&other.repr).unwrap(),
            ..Default::default()
        }
    }
    pub fn ge(&self, other: &Self) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.ge(&other.repr).unwrap(),
            ..Default::default()
        }
    }

    pub fn eq(&self, other: &Self) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.eq(&other.repr).unwrap(),
            ..Default::default()
        }
    }

    pub fn ne(&self, other: &Self) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.ne(&other.repr).unwrap(),
            ..Default::default()
        }
    }

    pub fn cmp(&self, other: &Self, ord: CmpOp) -> Tensor<S, u8, D> {
        Tensor {
            repr: self.repr.cmp(&other.repr, ord).unwrap(),
            ..Default::default()
        }
    }
}

impl<S: Shape, K: Kind, D: Device> PartialEq for Tensor<S, K, D> {
    fn eq(&self, other: &Self) -> bool {
        self.repr
            .data()
            .iter()
            .zip(other.repr.data().iter())
            .all(|(a, b)| a == b)
    }
}
