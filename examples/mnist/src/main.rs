#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use clap::Parser;
use kindle::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    train: bool,
    #[arg(short, long, default_value = "mnist.safetensors")]
    path: String,
}

dataset! {
    MnistTrainDataset,
    "/home/xupremix/Desktop/train.parquet"
}

dataset! {
    MnistTestDataset,
    "/home/xupremix/Desktop/test.parquet"
}

const LR: f64 = 5e-4;
const BATCH_SIZE: usize = 120;
const TEST_BATCH_SIZE: usize = 50;
const EPOCHS: usize = 3;
const N_SAMPLES: usize = 60_000;
const NORMALIZE: f32 = 255.;

struct MnistModel {
    conv1: Conv2d<1, 32>,
    maxp1: MaxPool2d<Kernel<2>, Stride<2>>,
    conv2: Conv2d<32, 64>,
    maxp2: MaxPool2d<Kernel<2>, Stride<2>>,
    fc: Linear<1600, 10>,
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
        train: bool,
    ) -> Tensor<Rank2<BATCH, 10>> {
        let xs = self.conv1.forward(xs).relu();
        let xs = self.maxp1.forward(&xs);
        let xs = self.conv2.forward(&xs).relu();
        let mut xs = self.maxp2.forward(&xs).flatten_from::<1, Rank2<_, _>>();
        if train {
            xs = xs.dropout(0.5);
        }
        self.fc.forward(&xs)
    }
}

fn main() {
    let Args { train, path } = Args::parse();

    if train {
        let vm = VarMap::new();
        let vs: Vs = Vs::from_varmap(&vm);
        let train_dataset: MnistTrainDataset = MnistTrainDataset::load().unwrap();
        let tensors = train_dataset.images.unsqueeze::<1>().to_kind::<f32>() / NORMALIZE;

        let x_train = &tensors.chunk::<0, BATCH_SIZE>();
        let y_train = &train_dataset
            .labels
            .to_kind::<u32>()
            .chunk::<0, BATCH_SIZE>();

        let model = MnistModel::new(&vs);
        let mut optim = AdamW::new(vm.all_vars(), LR);
        let mut acc = 0.;

        for epoch in 1..=EPOCHS {
            let mut total_loss = 0.;
            let mut n_correct = 0;

            for (x, y) in x_train.iter().zip(y_train) {
                let logits = model.forward(x, true);

                // Loss calculation
                let loss = Loss::cross_entropy(&logits, y);
                // println!("loss: {:.4}", loss.to_scalar());
                total_loss += loss.to_scalar();
                optim.backward_step(&loss);

                // Accuracy calculation
                let preds = logits.argmax::<1>();
                let correct = preds.eq(y).to_kind::<u32>().sum_all().to_scalar();
                n_correct += correct;
            }

            acc = n_correct as f32 / N_SAMPLES as f32;
            println!("epoch {epoch}: loss: {total_loss:.4}, accuracy: {acc:.4}");
        }
        vm.save(format!("mnist{:.0}.safetensors", acc * 100.))
            .unwrap();
    } else {
        // Testing with the trained model
        let mut vm = VarMap::new();
        vm.load(path).unwrap();
        let vs: Vs = Vs::from_varmap(&vm);
        let model = MnistModel::new(&vs);

        let test_dataset: MnistTestDataset = MnistTestDataset::load().unwrap();
        let samples = test_dataset.images.shape()[0];
        let x_test = test_dataset.images.unsqueeze::<1>().to_kind::<f32>() / NORMALIZE;

        let x_test = &x_test.chunk::<0, TEST_BATCH_SIZE>();
        let y_test = &test_dataset
            .labels
            .to_kind::<u32>()
            .chunk::<0, TEST_BATCH_SIZE>();

        let mut correct = 0;
        for (x_batch, y_batch) in x_test.iter().zip(y_test) {
            let preds = model
                .forward(&x_batch, false)
                .softmax_last_dim()
                .argmax::<1>();
            correct += preds.eq(&y_batch).to_kind::<u32>().sum_all().to_scalar();
        }

        let accuracy = correct as f32 / samples as f32;
        println!("Accuracy on test set: {accuracy:.4}");
    }
}
