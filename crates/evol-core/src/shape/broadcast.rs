use super::Shape;

pub trait Broadcast<Rhs: Shape, Dst: Shape>: Shape {
    const BROADCAST_CHECK: ();
}

pub trait BroadcastAs<Src: Shape>: Shape {
    const BROADCAST_AS_CHECK: ();
}

pub trait BroadcastLeft<Src: Shape>: Shape {
    type Extended: Shape;
}
