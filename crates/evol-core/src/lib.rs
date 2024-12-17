#![allow(private_bounds)]
#![allow(path_statements)]
#![allow(clippy::self_named_constructors)]
#![allow(incomplete_features)]
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

// TODO: ADD DOCUMENTATION

pub mod data;
pub mod device;
pub mod kind;
pub mod nn;
pub mod shape;
pub mod tensor;
pub(crate) mod utils;

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
