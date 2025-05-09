#![allow(private_bounds)]
#![allow(path_statements)]
#![allow(clippy::self_named_constructors)]
#![allow(incomplete_features)]
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

pub mod data;
pub mod device;
pub mod kind;
pub mod nn;
pub mod shape;
pub mod tensor;
pub(crate) mod utils;
pub use tch;

pub fn testing() {
    let t = tch::Tensor::rand([2, 2], tch::kind::FLOAT_CPU);
    let bytes = t.contiguous().data_ptr() as *const f32;
    let slice =
        unsafe { std::slice::from_raw_parts(bytes, t.numel() * std::mem::size_of::<f32>()) };
    let ct = candle_core::Tensor::from_slice(slice, (2, 2), &candle_core::Device::Cpu).unwrap();
    println!("{}", ct);
    t.print();

    // there is an alternative maybe using bytemuch and doing t.copy_data instead of from raw
    // parts, however that also requires an unsafe operation since we need to set manually the size
    // of a vector since we're copying data directly into it
}

pub mod prelude {
    use super::*;

    pub use evol_macros::*;

    pub use data::loader::*;
    pub use data::*;
    pub use device::*;
    pub use kind::*;
    pub use nn::prelude::*;
    pub use nn::*;
    pub use shape::*;
    pub use tensor::prelude::*;
    pub use tensor::*;
    pub use utils::CmpOp;
}

pub use evol_macros as macros;

pub mod candle {
    pub use candle_core as core;
    pub use candle_datasets as datasets;
    pub use candle_nn as nn;
    pub use candle_transformers as transformers;

    #[cfg(feature = "onnx")]
    pub use candle_onnx as onnx;
}
