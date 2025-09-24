use std::fmt::Debug;
use std::hash::Hash;

use kindle_macros::shape;

pub use candle_core::Shape as CandleShape;

use crate::nn::Forward;
use crate::tensor::method_traits::*;

use crate::kind::Kind;

pub trait Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash {
    type Shape<K: Kind>: 'static + Clone + Copy + Send + Sync + PartialEq;
    const DIMS: usize;
    const NELEMS: usize;
    fn shape() -> CandleShape;
    fn dims() -> &'static [usize];
    fn as_slice<K: Kind>(shape: &Self::Shape<K>) -> &[K];
}

shape!(pub 0@);
shape!(pub 1@);
shape!(pub 2@);
shape!(pub 3@);
shape!(pub 4@);
shape!(pub 5@);
shape!(pub 6@);
shape!(pub 7@);
shape!(pub 8@);
