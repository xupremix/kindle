use super::Shape;

pub trait Broadcast<Rhs: Shape, Dst: Shape>: Shape {
    const BROADCAST_CHECK: ();
}
