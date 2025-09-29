#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use kindle::{nn::optim::Backward, prelude::*};

dataset! {
    MnistDataset,
    "/home/xupremix/Desktop/train.parquet"
}

const LR: f64 = 0.1;
const BATCH_SIZE: usize = 125;
const EPOCHS: usize = 15;
const N_SAMPLES: usize = 60_000;

struct MnistModel {
    conv1: Conv2d<1, 32>,
    maxp1: MaxPool2d<Kernel<2>, Stride<2>>,
    conv2: Conv2d<32, 64>,
    maxp2: MaxPool2d<Kernel<2>, Stride<2>>,
    fc: Linear<1_600, 10>,
}

impl MnistModel {
    fn new(vs: &Vs) -> Self {
        Self {
            conv1: Conv2d::conv2d(vs, "conv1"),
            maxp1: MaxPool2d::new(),
            conv2: Conv2d::conv2d(vs, "conv2"),
            maxp2: MaxPool2d::new(),
            fc: Linear::linear(vs, "fc"),
        }
    }

    fn forward<const BATCH: usize>(
        &self,
        xs: &Tensor<Rank4<BATCH, 1, 28, 28>>,
    ) -> Tensor<Rank2<BATCH, 10>> {
        let xs = self.conv1.forward(xs).relu();
        let xs = self.maxp1.forward(&xs);
        let xs = self.conv2.forward(&xs).relu();
        let xs = self
            .maxp2
            .forward(&xs)
            .flatten_from::<1, Rank2<_, _>>()
            .dropout(0.5);
        self.fc.forward(&xs).softmax::<1>()
    }
}

fn main() {
    let dataset: MnistDataset = MnistDataset::load().unwrap();
    let tensors = dataset.images.unsqueeze::<1>().to_kind::<f32>();

    let x_train = &tensors.chunk::<0, BATCH_SIZE>();
    let y_train = &dataset.labels.to_kind::<u32>().chunk::<0, BATCH_SIZE>();

    // let x_test = x_train.pop();
    // let y_test = y_train.pop();

    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let model = MnistModel::new(&vs);
    let mut optim = AdamW::new(vm.all_vars(), LR);

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.;
        let mut n_correct = 0;

        for (x, y) in x_train.iter().zip(y_train) {
            let logits = model.forward(x);

            // Loss calculation
            let loss = Loss::cross_entropy(&logits, y);
            total_loss += loss.to_scalar();
            optim.backward_step(&loss);

            // Accuracy calculation
            let preds = logits.argmax::<1>();
            let correct = preds.eq(y).to_kind::<u32>().sum_all().to_scalar();
            n_correct += correct;
        }

        let acc = n_correct as f32 / N_SAMPLES as f32;
        println!("epoch {epoch}: loss: {total_loss:.4}, accuracy: {acc:.4}")
    }

    vm.save("./model_save.safetensors").unwrap();
}
