use crate::utils::{ToUsize2, Window};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxPool2D<Pool: ToUsize2 + 'static = Window<3>, Stride: ToUsize2 + 'static = Window<1>> {
    __pool: PhantomData<Pool>,
    __stride: PhantomData<Stride>,
}
