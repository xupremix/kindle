use std::fmt::Debug;

pub trait DType: 'static + Debug + Clone + Copy + Send + Sync + PartialEq {
    fn dtype() -> candle_core::DType;
}

macro_rules! dtype {
    ($( $t:ty, $n:ident );* $(;)?) => {
        $(
            impl DType for $t {
                fn dtype() -> candle_core::DType {
                    candle_core::DType::$n
                }
            }
        )*
    };
}

dtype! {
    u8, U8;
    u32, U32;
    i64, I64;
    f32, F32;
    f64, F64;
}

#[cfg(feature = "half")]
mod custom {
    use super::*;

    use half::f16;

    #[allow(non_camel_case_types)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct bf16;

    dtype! {
        f16, F16;
        bf16, BF16;
    }
}

#[cfg(feature = "half")]
pub use custom::bf16;
