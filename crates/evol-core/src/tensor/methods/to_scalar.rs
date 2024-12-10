use crate::{
    prelude::ToScalar,
    shape::Scalar,
    tensor::{Device, Kind, Tensor},
};

impl<K: Kind, D: Device> Tensor<Scalar, K, D> {
    pub fn to_scalar(&self) -> K {
        Scalar::TO_SCALAR_CHECK;
        self.repr.to_scalar().unwrap()
    }
}
