use crate::tensor::{Device, Kind, Shape, Tensor};

macro_rules! wrap {
    ($( $f:ident )*) => {
        impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
            $(
                pub fn $f(&self) -> Self {
                    Self {
                        repr: self.repr.$f().unwrap(),
                        ..Default::default()
                    }
                }
            )*
        }
    };
}

wrap! {
    abs
    cos
    erf
    floor
    gelu
    gelu_erf
    log
    relu
    round
    silu
    sin
    sqr
    sqrt
    tanh
}
