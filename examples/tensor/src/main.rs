use evol::{candle::candle_core, prelude::*};

fn main() {
    let t1 = candle_core::Tensor::ones((3,), f32::DTYPE, &Cpu::device()).unwrap();
    let t2 = candle_core::Tensor::ones((3, 4), f32::DTYPE, &Cpu::device()).unwrap();
    let ris = t1.broadcast_matmul(&t2).unwrap();
    println!("{}", ris);
}
