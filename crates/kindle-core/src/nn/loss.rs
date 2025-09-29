// mse, cross_entropy, cross_entropy_with_logits, nll

use candle_core::WithDType;
use candle_nn::loss::cross_entropy;

use crate::{
    device::Device,
    shape::{Rank1, Rank2, Scalar},
    tensor::Tensor,
};

pub struct Loss;

impl Loss {
    pub fn cross_entropy<
        const BATCH_SIZE: usize,
        const CATEGORIES: usize,
        K: WithDType,
        D: Device,
    >(
        input: &Tensor<Rank2<BATCH_SIZE, CATEGORIES>, K, D>,
        target: &Tensor<Rank1<BATCH_SIZE>, u32, D>,
    ) -> Tensor<Scalar, K, D> {
        Tensor {
            repr: cross_entropy(&input.repr, &target.repr).unwrap(),
            ..Default::default()
        }
    }
}
