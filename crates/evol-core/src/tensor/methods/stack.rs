use crate::{
    prelude::Stack,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn stack<const DIM: usize, const N: usize>(
        tensors: &[&Self; N],
    ) -> Tensor<S::StackShape, K, D>
    where
        S: Stack<DIM, N>,
    {
        let tensors = tensors.iter().map(|&t| &t.repr).collect::<Vec<_>>();
        Tensor {
            repr: candle_core::Tensor::stack(&tensors, DIM).unwrap(),
            ..Default::default()
        }
    }
}
