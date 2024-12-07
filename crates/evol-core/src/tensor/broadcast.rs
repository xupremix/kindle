use crate::shape::broadcast::Broadcast;
use crate::tensor::{Device, Kind, Shape, Tensor};

// impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
//     pub fn broadcast<Dst: Shape, Rhs: Broadcast<S, Dst>>(&self) -> Tensor<Dst, K, D> {
//         Rhs::BROADCAST_CHECK;
//         Tensor {
//             repr: self.repr.broadcast_as(Dst::shape()).unwrap(),
//             ..Default::default()
//         }
//     }
// }

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
