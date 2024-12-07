// use std::ops::{Add, Div, Sub};
//
// use crate::shape::Rank1;
// use crate::tensor::{Device, Kind, Shape, Tensor};
//
// trait Arange<
//     K: Kind + Sub<K> + Add<K> + Div<K>,
//     const SIZE: K,
//     const END: K,
//     const START: K,
//     const STEP: K,
// >
// {
//     const CHECK: () = assert!(
//         SIZE == (END - START) / STEP,
//         "Size provided must be equal to (end - start) / step"
//     );
// }
// impl<K: Kind, const SIZE: K, const END: K, const START: K, const STEP: K>
//     Arange<SIZE, END, START, STEP> for ()
// {
// }
//
// impl<const SIZE: usize, K: Kind, D: Device> Tensor<Rank1<SIZE>, K, D> {
//     pub fn arange<const START: K, const END: K, const STEP: K>() -> Self {
//         <() as Arange<SIZE, END, START, STEP>>::CHECK;
//         Self {
//             repr: candle_core::Tensor::arange(START, END, STEP, &D::device()).unwrap(),
//             ..Default::default()
//         }
//     }
// }
