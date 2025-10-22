use crate::{
    prelude::Cat,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn cat<const DIM: usize, const N: usize, Dst: Shape>(
        tensors: &[impl AsRef<Self>; N],
    ) -> Tensor<Dst, K, D>
    where
        S: Cat<DIM, N, Dst>,
    {
        S::CAT_CHECK;
        let tensors = tensors.iter().map(|t| &t.as_ref().repr).collect::<Vec<_>>();
        Tensor {
            repr: candle_core::Tensor::cat(&tensors, DIM).unwrap(),
            ..Default::default()
        }
    }
}
