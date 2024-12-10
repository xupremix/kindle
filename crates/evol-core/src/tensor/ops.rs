use crate::kind::ToF64;
use crate::{device::Device, kind::Kind, shape::Shape, tensor::Tensor};

macro_rules! op {
    ($( $name:ident $fn:ident $op:tt)*) => {
        $(
            // Scalar ops

            impl<S: Shape, K: Kind, D: Device, T: ToF64> std::ops::$name<T> for Tensor<S, K, D> {
                type Output = Tensor<S, K, D>;
                #[inline(always)]
                fn $fn(self, rhs: T) -> Self::Output {
                    Tensor {
                        repr: ( self.repr $op rhs.to_f64() ).unwrap(),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, K: Kind, D: Device, T: ToF64> std::ops::$name<T> for &Tensor<S, K, D> {
                type Output = Tensor<S, K, D>;
                #[inline(always)]
                fn $fn(self, rhs: T) -> Self::Output {
                    Tensor {
                        repr: ( &self.repr $op rhs.to_f64() ).unwrap(),
                        ..Default::default()
                    }
                }
            }

            // Tensor ops

            impl<S: Shape, K: Kind, D: Device> std::ops::$name for Tensor<S, K, D> {
                type Output = Tensor<S, K, D>;
                #[inline(always)]
                fn $fn(self, rhs: Self) -> Self::Output {
                    Tensor {
                        repr: ( self.repr $op rhs.repr ).unwrap(),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, K: Kind, D: Device> std::ops::$name<&Self> for Tensor<S, K, D> {
                type Output = Tensor<S, K, D>;
                #[inline(always)]
                fn $fn(self, rhs: &Self) -> Self::Output {
                    Tensor {
                        repr: ( self.repr $op &rhs.repr ).unwrap(),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, K: Kind, D: Device> std::ops::$name for &Tensor<S, K, D> {
                type Output = Tensor<S, K, D>;
                #[inline(always)]
                fn $fn(self, rhs: Self) -> Self::Output {
                    Tensor {
                        repr: ( &self.repr $op &rhs.repr ).unwrap(),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, K: Kind, D: Device> std::ops::$name<Tensor<S, K, D>> for &Tensor<S, K, D> {
                type Output = Tensor<S, K, D>;
                #[inline(always)]
                fn $fn(self, rhs: Tensor<S, K, D>) -> Self::Output {
                    Tensor {
                        repr: ( &self.repr $op rhs.repr ).unwrap(),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

op! {
    Add add +
    Sub sub -
    Mul mul *
    Div div /
}
