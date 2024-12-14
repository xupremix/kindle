use crate::{
    prelude::Narrow,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn narrow<const DIM: usize, const START: usize, const LEN: usize>(
        &self,
    ) -> Tensor<S::NarrowShape, K, D>
    where
        S: Narrow<DIM, START, LEN>,
    {
        S::NARROW_CHECK;
        Tensor {
            repr: self.repr.narrow(DIM, START, LEN).unwrap(),
            ..Default::default()
        }
    }
}
