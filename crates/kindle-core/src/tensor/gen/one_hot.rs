use candle_core::op::CmpOp;

use crate::err::{KindleError, KindleResult};
use crate::shape::Rank1;
use crate::tensor::{Device, Kind, Tensor};

struct Tmp;

trait OneHot<const DIM: usize, const POS: usize> {
    const CHECK: () = assert!(
        POS < DIM,
        "The position of the hot encoded 1 is out of bounds\n"
    );
}

impl<const DIM: usize, const POS: usize> OneHot<DIM, POS> for Tmp {}

impl<const DIM: usize, K: Kind, D: Device> Tensor<Rank1<DIM>, K, D> {
    pub fn one_hot<const POS: usize>() -> Self {
        <Tmp as OneHot<DIM, POS>>::CHECK;
        Tensor::arange().cmp(POS as u32, CmpOp::Eq).to_kind()
    }

    pub fn dyn_one_hot(pos: u32) -> KindleResult<Self> {
        if pos as usize >= DIM {
            Err(KindleError::DimensionOutOfBounds {
                idx: pos as usize,
                length: DIM,
            })
        } else {
            Ok(Tensor::arange().cmp(pos, CmpOp::Eq).to_kind())
        }
    }
}
