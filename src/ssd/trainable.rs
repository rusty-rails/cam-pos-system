use super::dataset::DataSet;
use super::lenet::Lenet;
use super::mobilenet::MobileNet;
use async_trait::async_trait;
use autograph::{
    device::Device,
    learn::{
        neural_network::{ NetworkTrainer},
        Summarize, Test, Train,
    },
    result::Result,
};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use rand::seq::SliceRandom;
use std::time::Instant;
use std::{ fs, path::PathBuf};

macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

fn progress_iter<X>(
    iter: impl ExactSizeIterator<Item = X>,
    epoch: usize,
    name: &str,
) -> impl ExactSizeIterator<Item = X> {
    let style = ProgressStyle::default_bar()
        .template(&format!(
            "[epoch: {} elapsed: {{elapsed}}] {} [{{bar}}] {{pos:>7}}/{{len:7}} [eta: {{eta}}]",
            epoch, name
        ))
        .unwrap()
        .progress_chars("=> ");
    let bar = ProgressBar::new(iter.len() as u64).with_style(style);
    iter.progress_with(bar)
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

#[async_trait]
pub trait Trainable {
    async fn train(&mut self, dataset: &DataSet, epochs: usize) -> Result<()>;
    fn evaluate(&self, dataset: &DataSet);
}

#[async_trait]
impl Trainable for Lenet {
    async fn train(&mut self, dataset: &DataSet, epochs: usize) -> Result<()> {
        let ((x_train, y_train), (x_test, y_test)) = dataset.get();
        let (x_train, y_train, x_test, y_test) =
            (x_train.view(), y_train.view(), x_test.view(), y_test.view());
        let batch_size = 64usize;
        let device = Device::new().unwrap();

        let save_path: Option<PathBuf> = None;
        // Construct a trainer to train the network.
        let mut trainer = match save_path.as_ref() {
            // Load the trainer from a file.
            Some(save_path) if save_path.exists() => bincode::deserialize(&fs::read(save_path)?)?,
            // Use the provided layer.
            _ => NetworkTrainer::from_network(self.net.clone()),
        };

        trainer.to_device_mut(device.clone()).await?;
        println!("{:#?}", &trainer);

        for epoch in 0..epochs {
            let train_iter = progress_iter(
                DataSet::shuffled_batch_iter(&device, &x_train, &y_train, batch_size),
                epoch,
                "training",
            );
            let test_iter = progress_iter(
                DataSet::batch_iter(&device, &x_test, &y_test, batch_size),
                epoch,
                "testing",
            );
            trainer.train_test(train_iter, test_iter)?;
            println!("{:#?}", trainer.summarize());

            // Save the trainer at each epoch.
            if let Some(save_path) = save_path.as_ref() {
                fs::write(save_path, bincode::serialize(&trainer)?)?;
            }
        }

        println!("Evaluating...");
        let test_iter = progress_iter(
            DataSet::batch_iter(&device, &x_test, &y_test, batch_size),
            trainer.summarize().epoch,
            "evaluating",
        );
        let stats = trainer.test(test_iter)?;
        println!("{:#?}", stats);

        Ok(())
    }

    fn evaluate(&self, dataset: &DataSet) {
        let device = Device::new().unwrap();
        let ((_x_train, _y_train), (x_test, y_test)) = dataset.get();
    }
}

#[async_trait]
impl Trainable for MobileNet {
    async fn train(&mut self, dataset: &DataSet, epochs: usize) -> Result<()> {
        let ((x_train, y_train), (x_test, y_test)) = dataset.get();
        let (x_train, y_train, x_test, y_test) =
            (x_train.view(), y_train.view(), x_test.view(), y_test.view());
        let batch_size = 64usize;
        let device = Device::new().unwrap();

        let save_path: Option<PathBuf> = None;
        // Construct a trainer to train the network.
        let mut trainer = match save_path.as_ref() {
            // Load the trainer from a file.
            Some(save_path) if save_path.exists() => bincode::deserialize(&fs::read(save_path)?)?,
            // Use the provided layer.
            _ => NetworkTrainer::from_network(self.net.clone().into()),
        };

        trainer.to_device_mut(device.clone()).await?;
        println!("{:#?}", &trainer);

        for epoch in 0..epochs {
            let train_iter = progress_iter(
                DataSet::shuffled_batch_iter(&device, &x_train, &y_train, batch_size),
                epoch,
                "training",
            );
            let test_iter = progress_iter(
                DataSet::batch_iter(&device, &x_test, &y_test, batch_size),
                epoch,
                "testing",
            );
            trainer.train_test(train_iter, test_iter)?;
            println!("{:#?}", trainer.summarize());

            // Save the trainer at each epoch.
            if let Some(save_path) = save_path.as_ref() {
                fs::write(save_path, bincode::serialize(&trainer)?)?;
            }
        }

        println!("Evaluating...");
        let test_iter = progress_iter(
            DataSet::batch_iter(&device, &x_test, &y_test, batch_size),
            trainer.summarize().epoch,
            "evaluating",
        );
        let stats = trainer.test(test_iter)?;
        println!("{:#?}", stats);

        Ok(())
    }

    fn evaluate(&self, dataset: &DataSet) {
        let ((_x_train, _y_train), (x_test, y_test)) = dataset.get();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LABELS: usize = 18;
    const IMAGES_PER_LABEL: usize = 21;

    #[tokio::test]
    async fn test_training() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), 18);
        let mut model = Lenet::new(28, 28);
        model.train(&dataset, 10).await.unwrap();
    }

    #[test]
    fn test_evaluate() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
        let mut model = Lenet::new(28, 28);
        model.train(&dataset, 100);
        model.evaluate(&dataset);
    }
}
