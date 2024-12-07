pub use candle_core::WithDType as Kind;

#[cfg(feature = "half")]
pub use half::{bf16, f16};

pub(crate) trait ToF64 {
    fn to_f64(self) -> f64;
}

macro_rules! to_f64 {
    ($($t:ty)*) => {
        $(
            impl ToF64 for $t {
                #[inline(always)]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

to_f64! {
    u8 u16 u32 u64 u128 usize
    i8 i16 i32 i64 i128 isize
    f32 f64
}

#[cfg(feature = "half")]
impl ToF64 for f16 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
#[cfg(feature = "half")]
impl ToF64 for bf16 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
