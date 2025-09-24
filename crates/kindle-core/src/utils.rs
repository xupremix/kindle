pub(crate) trait Sealed {}
pub use candle_core::op::CmpOp;

pub trait ToUsize2 {
    const FIRST: usize;
    const SECOND: usize;
    const TUPLE: (usize, usize) = (Self::FIRST, Self::SECOND);
}

pub struct Window<const FIRST: usize, const SECOND: usize = FIRST>;
impl<const FIRST: usize, const SECOND: usize> ToUsize2 for Window<FIRST, SECOND> {
    const FIRST: usize = FIRST;
    const SECOND: usize = SECOND;
}

pub type Kernel<const FIRST: usize, const SECOND: usize = FIRST> = Window<FIRST, SECOND>;
pub type Pool<const FIRST: usize, const SECOND: usize = FIRST> = Window<FIRST, SECOND>;
pub type Stride<const FIRST: usize, const SECOND: usize = FIRST> = Window<FIRST, SECOND>;
