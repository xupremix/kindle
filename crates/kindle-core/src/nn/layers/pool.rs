use candle_core::WithDType;

use crate::device::Device;
use crate::nn::Module;
use crate::shape::Rank4;
use crate::tensor::{FromCandleTensor, Tensor};
use crate::utils::{ToUsize2, Window};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxPool2D<Kernel: ToUsize2 + 'static = Window<3>, Stride: ToUsize2 + 'static = Window<1>>
{
    __pool: PhantomData<Kernel>,
    __stride: PhantomData<Stride>,
}

impl Default for MaxPool2D<Window<3>, Window<1>> {
    fn default() -> Self {
        Self {
            __pool: PhantomData,
            __stride: PhantomData,
        }
    }
}

impl<Kernel: ToUsize2 + 'static, Stride: ToUsize2 + 'static> MaxPool2D<Kernel, Stride> {
    pub fn new() -> Self {
        Self {
            __pool: PhantomData,
            __stride: PhantomData,
        }
    }
}

impl<
        const N: usize,
        const CIN: usize,
        const HIN: usize,
        const WIN: usize,
        K: WithDType,
        D: Device,
        Kernel: ToUsize2,
        Stride: ToUsize2,
    > Module<Tensor<Rank4<N, CIN, HIN, WIN>, K, D>> for MaxPool2D<Kernel, Stride>
where
    [(); f32::floor((HIN - Kernel::FIRST) as f32 / Stride::FIRST as f32) as usize + 1]:,
    [(); f32::floor((WIN - Kernel::SECOND) as f32 / Stride::SECOND as f32) as usize + 1]:,
{
    type Output = Tensor<
        Rank4<
            N,
            CIN,
            { f32::floor((HIN - Kernel::FIRST) as f32 / Stride::FIRST as f32) as usize + 1 },
            { f32::floor((WIN - Kernel::SECOND) as f32 / Stride::SECOND as f32) as usize + 1 },
        >,
    >;

    fn forward(&self, xs: &Tensor<Rank4<N, CIN, HIN, WIN>, K, D>) -> Self::Output {
        Tensor::from_candle_tensor(
            xs.repr
                .max_pool2d_with_stride(Kernel::TUPLE, Stride::TUPLE)
                .unwrap(),
        )
    }
}

impl<Kernel: ToUsize2 + Default, Stride: ToUsize2 + Default> MaxPool2D<Kernel, Stride> {
    pub fn forward<
        const N: usize,
        const CIN: usize,
        const HIN: usize,
        const WIN: usize,
        K: WithDType,
        D: Device,
    >(
        xs: &Tensor<Rank4<N, CIN, HIN, WIN>, K, D>,
    ) -> <Self as Module<Tensor<Rank4<N, CIN, HIN, WIN>, K, D>>>::Output
    where
        Self: Module<Tensor<Rank4<N, CIN, HIN, WIN>, K, D>>,
    {
        Self::new().forward(xs)
    }
}
