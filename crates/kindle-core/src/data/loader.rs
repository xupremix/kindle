use std::marker::PhantomData;

use crate::{
    device::Device,
    shape::{Rank1, Rank2, Rank4},
    tensor::{FromCandleTensor as _, Tensor},
};

#[cfg(feature = "cuda")]
use crate::device::Cuda;

#[cfg(not(feature = "cuda"))]
use crate::device::Cpu;

use super::Dataset;

pub struct DataLoader<
    #[cfg(not(feature = "cuda"))] D: Device = Cpu,
    #[cfg(feature = "cuda")] D: Device = Cuda,
> {
    __device: PhantomData<D>,
}

type Mnist<D> = Dataset<
    10,
    Tensor<Rank2<60000, 784>, f32, D>,
    Tensor<Rank1<60000>, u8, D>,
    Tensor<Rank2<10000, 784>, f32, D>,
    Tensor<Rank1<10000>, u8, D>,
>;

type Cifar<D> = Dataset<
    10,
    Tensor<Rank4<50000, 3, 32, 32>, u8, D>,
    Tensor<Rank1<50000>, u8, D>,
    Tensor<Rank4<10000, 3, 32, 32>, u8, D>,
    Tensor<Rank1<10000>, u8, D>,
>;

impl<D: Device> DataLoader<D> {
    pub fn mnist() -> Mnist<D> {
        let dt = candle_datasets::vision::mnist::load().unwrap();
        Mnist {
            train_images: Tensor::from_candle_tensor(
                dt.train_images.to_device(&D::device()).unwrap(),
            ),
            train_labels: Tensor::from_candle_tensor(
                dt.train_labels.to_device(&D::device()).unwrap(),
            ),
            test_images: Tensor::from_candle_tensor(
                dt.test_images.to_device(&D::device()).unwrap(),
            ),
            test_labels: Tensor::from_candle_tensor(
                dt.test_labels.to_device(&D::device()).unwrap(),
            ),
        }
    }

    pub fn cifar() -> Cifar<D> {
        let dt = candle_datasets::vision::cifar::load().unwrap();
        Cifar {
            train_images: Tensor::from_candle_tensor(
                dt.train_images.to_device(&D::device()).unwrap(),
            ),
            train_labels: Tensor::from_candle_tensor(
                dt.train_labels.to_device(&D::device()).unwrap(),
            ),
            test_images: Tensor::from_candle_tensor(
                dt.test_images.to_device(&D::device()).unwrap(),
            ),
            test_labels: Tensor::from_candle_tensor(
                dt.test_labels.to_device(&D::device()).unwrap(),
            ),
        }
    }
}
