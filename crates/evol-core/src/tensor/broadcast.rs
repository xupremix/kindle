use crate::prelude::{BroadcastAs, BroadcastLeft};
use crate::shape::broadcast::Broadcast;
use crate::tensor::{Device, Kind, Shape, Tensor};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn broadcast<Dst: BroadcastAs<S>>(&self) -> Tensor<Dst, K, D> {
        Dst::BROADCAST_AS_CHECK;
        Tensor {
            repr: self.repr.broadcast_as(Dst::shape()).unwrap(),
            ..Default::default()
        }
    }
}

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn broadcast_left<L: BroadcastLeft<S>>(&self) -> Tensor<L::Extended, K, D> {
        Tensor {
            repr: self.repr.broadcast_left(L::shape()).unwrap(),
            ..Default::default()
        }
    }
}

macro_rules! op {
    ($( $name:ident )*) => {
        $(
            impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
                pub fn $name<Dst: Shape, Rhs: Broadcast<S, Dst>>(
                    &self,
                    rhs: &Tensor<Rhs, K, D>,
                ) -> Tensor<Dst, K, D> {
                    Rhs::BROADCAST_CHECK;
                    Tensor {
                        repr: self.repr.$name(&rhs.repr).unwrap(),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

op! {
    broadcast_add
    broadcast_div
    broadcast_mul
    broadcast_sub
}

macro_rules! op_cmp {
    ($( $name:ident )*) => {
        $(
            impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
                pub fn $name<Dst: Shape, Rhs: Broadcast<S, Dst>>(
                    &self,
                    rhs: &Tensor<Rhs, K, D>,
                ) -> Tensor<Dst, u8, D> {
                    Rhs::BROADCAST_CHECK;
                    Tensor {
                        repr: self.repr.$name(&rhs.repr).unwrap(),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

op_cmp! {
    broadcast_eq
    broadcast_ge
    broadcast_gt
    broadcast_le
    broadcast_lt
    broadcast_ne
}
