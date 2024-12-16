use super::Module;

use crate::prelude::{Device, Kind, Shape, Tensor};

impl<S, K, D, M0, M1> Module<Tensor<S, K, D>> for (M0, M1)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
{
    type Output = M1::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        self.1.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2> Module<Tensor<S, K, D>> for (M0, M1, M2)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
{
    type Output = M2::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        self.2.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2, M3> Module<Tensor<S, K, D>> for (M0, M1, M2, M3)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
{
    type Output = M3::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        let xs = self.2.forward(&xs);
        self.3.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2, M3, M4> Module<Tensor<S, K, D>> for (M0, M1, M2, M3, M4)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
    M4: Module<M3::Output>,
{
    type Output = M4::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        let xs = self.2.forward(&xs);
        let xs = self.3.forward(&xs);
        self.4.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2, M3, M4, M5> Module<Tensor<S, K, D>> for (M0, M1, M2, M3, M4, M5)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
    M4: Module<M3::Output>,
    M5: Module<M4::Output>,
{
    type Output = M5::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        let xs = self.2.forward(&xs);
        let xs = self.3.forward(&xs);
        let xs = self.4.forward(&xs);
        self.5.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2, M3, M4, M5, M6> Module<Tensor<S, K, D>> for (M0, M1, M2, M3, M4, M5, M6)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
    M4: Module<M3::Output>,
    M5: Module<M4::Output>,
    M6: Module<M5::Output>,
{
    type Output = M6::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        let xs = self.2.forward(&xs);
        let xs = self.3.forward(&xs);
        let xs = self.4.forward(&xs);
        let xs = self.5.forward(&xs);
        self.6.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2, M3, M4, M5, M6, M7> Module<Tensor<S, K, D>>
    for (M0, M1, M2, M3, M4, M5, M6, M7)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
    M4: Module<M3::Output>,
    M5: Module<M4::Output>,
    M6: Module<M5::Output>,
    M7: Module<M6::Output>,
{
    type Output = M7::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        let xs = self.2.forward(&xs);
        let xs = self.3.forward(&xs);
        let xs = self.4.forward(&xs);
        let xs = self.5.forward(&xs);
        let xs = self.6.forward(&xs);
        self.7.forward(&xs)
    }
}
impl<S, K, D, M0, M1, M2, M3, M4, M5, M6, M7, M8> Module<Tensor<S, K, D>>
    for (M0, M1, M2, M3, M4, M5, M6, M7, M8)
where
    S: Shape,
    K: Kind,
    D: Device,
    M0: Module<Tensor<S, K, D>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
    M4: Module<M3::Output>,
    M5: Module<M4::Output>,
    M6: Module<M5::Output>,
    M7: Module<M6::Output>,
    M8: Module<M7::Output>,
{
    type Output = M8::Output;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        let xs = self.2.forward(&xs);
        let xs = self.3.forward(&xs);
        let xs = self.4.forward(&xs);
        let xs = self.5.forward(&xs);
        let xs = self.6.forward(&xs);
        let xs = self.7.forward(&xs);
        self.8.forward(&xs)
    }
}
