pub mod adam;
pub mod sgd;

pub trait Backward<T> {
    const BACKWARD_CHECK: ();
    fn backward_step(&mut self, loss: &T);
}
