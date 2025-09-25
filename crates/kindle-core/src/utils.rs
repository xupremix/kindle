pub(crate) trait Sealed {}
pub use candle_core::op::CmpOp;

pub trait ToUsize2 {
    const FIRST: usize;
    const SECOND: usize;
    const TUPLE: (usize, usize) = (Self::FIRST, Self::SECOND);
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Window<const H: usize, const W: usize = H>;
impl<const H: usize, const W: usize> ToUsize2 for Window<H, W> {
    const FIRST: usize = H;
    const SECOND: usize = W;
}

pub type Kernel<const H: usize, const W: usize = H> = Window<H, W>;
pub type Stride<const H: usize, const W: usize = H> = Window<H, W>;
