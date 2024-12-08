use crate::{
    prelude::{Flatten, FlattenAll, FlattenFrom},
    shape::Rank1,
    tensor::{Device, Kind, Shape, Tensor},
};

// flatten_all [0 - n]
// flatten_dim [x - x]
// flatten [start - end]
// flatten_to [0 - end]
// flatten_from [start - n]

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn flatten_all<const N: usize>(&self) -> Tensor<Rank1<N>, K, D>
    where
        S: FlattenAll<N>,
    {
        S::FLATTEN_ALL_CHECK;
        Tensor {
            repr: self.repr.flatten_all().unwrap(),
            ..Default::default()
        }
    }

    pub fn flatten_dim<const DIM: usize, Dst: Flatten<S, DIM, DIM>>(&self) -> Tensor<Dst, K, D> {
        Dst::FLATTEN_CHECK;
        Tensor {
            repr: self.repr.flatten(DIM, DIM).unwrap(),
            ..Default::default()
        }
    }

    pub fn flatten<const START: usize, const END: usize, Dst: Flatten<S, START, END>>(
        &self,
    ) -> Tensor<Dst, K, D> {
        Dst::FLATTEN_CHECK;
        Tensor {
            repr: self.repr.flatten(START, END).unwrap(),
            ..Default::default()
        }
    }

    pub fn flatten_to<const END: usize, Dst: Flatten<S, 0, END>>(&self) -> Tensor<Dst, K, D> {
        Dst::FLATTEN_CHECK;
        Tensor {
            repr: self.repr.flatten_to(END).unwrap(),
            ..Default::default()
        }
    }

    pub fn flatten_from<const START: usize, Dst: FlattenFrom<S, START>>(
        &self,
    ) -> Tensor<Dst, K, D> {
        Dst::FLATTEN_CHECK;
        Tensor {
            repr: self.repr.flatten_from(START).unwrap(),
            ..Default::default()
        }
    }
}
