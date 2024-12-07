use std::fmt::Debug;

use candle_core::WithDType;

pub(crate) mod conversions;

pub trait Kind: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + WithDType {
    fn kind() -> candle_core::DType;
}

macro_rules! kind {
    ($( $t:ty, $n:ident );* $(;)?) => {
        $(
            impl Kind for $t {
                fn kind() -> candle_core::DType {
                    candle_core::DType::$n
                }
            }
        )*
    };
}

kind! {
    u8, U8;
    u32, U32;
    i64, I64;
    f32, F32;
    f64, F64;
}

#[cfg(feature = "half")]
pub use half::{bf16, f16};
#[cfg(feature = "half")]
kind! {
    f16, F16;
    bf16, BF16;
}
