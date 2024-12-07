#![allow(private_bounds)]
#![allow(path_statements)]
#![allow(unused_imports)]

pub mod device;
pub mod kind;
pub mod nn;
pub mod shape;
pub mod tensor;
pub(crate) mod utils;

pub mod prelude {
    use super::*;

    pub use device::*;
    pub use evol_macros::*;
    pub use kind::*;
    pub use nn::*;
    pub use shape::broadcast::*;
    pub use shape::*;
    pub use tensor::methods::*;
    pub use tensor::*;
    pub use utils::CmpOp;
}

pub use evol_macros::*;

pub mod candle {
    pub use candle_core;
    pub use candle_nn;
    pub use candle_onnx;
    pub use candle_transformers;
}
