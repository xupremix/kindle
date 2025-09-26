use crate::shape::Rank1;
use crate::tensor::{Device, Kind, Tensor};

struct Tmp;

trait ArangeCheck<const SIZE: usize, const END: usize, const START: usize, const STEP: usize> {
    const CHECK: () = assert!(
        SIZE == f32::floor((END - START) as f32 / STEP as f32) as usize,
        "Size provided must be equal to (end - start) / step"
    );
}
impl<const SIZE: usize, const END: usize, const START: usize, const STEP: usize>
    ArangeCheck<SIZE, END, START, STEP> for Tmp
{
}

impl<const SIZE: usize, K: Kind, D: Device> Tensor<Rank1<SIZE>, K, D> {
    pub fn arange() -> Self {
        Self::arange_with::<0, SIZE, 1>()
    }

    pub fn arange_with<const START: usize, const END: usize, const STEP: usize>() -> Self {
        <Tmp as ArangeCheck<SIZE, END, START, STEP>>::CHECK;
        let repr =
            candle_core::Tensor::arange_step(START as u32, END as u32, STEP as u32, &D::device())
                .unwrap()
                .to_dtype(K::DTYPE)
                .unwrap();
        Self {
            repr,
            ..Default::default()
        }
    }
}
