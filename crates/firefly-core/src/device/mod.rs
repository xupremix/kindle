use crate::utils::Sealed;
use std::fmt::Debug;
use std::hash::Hash;

pub trait Device:
    'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash + Sealed
{
    fn device() -> candle_core::Device;
}

// Cpu device

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cpu;
impl Sealed for Cpu {}

impl Device for Cpu {
    fn device() -> candle_core::Device {
        candle_core::Device::Cpu
    }
}

// Cuda device

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Cuda<const N: usize = 0>;
    impl<const N: usize> Sealed for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        fn device() -> candle_core::Device {
            candle_core::Device::new_cuda(N).expect("Invalid cuda device")
        }
    }
}
#[cfg(feature = "cuda")]
pub use cuda::Cuda;

// Metal device

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Metal<const N: usize = 0>;
impl<const N: usize> Sealed for Metal<N> {}

impl<const N: usize> Device for Metal<N> {
    fn device() -> candle_core::Device {
        candle_core::Device::new_metal(N).expect("Invalid metal device")
    }
}
