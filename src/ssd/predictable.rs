use super::model::Model;
use super::trainable::compute_logits;
use ag::ndarray;
use ag::tensor_ops as T;
use autograd as ag;
use image::GrayImage;
use mosse::utils::preprocess;

pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

pub trait Predictable {
    fn predict(&mut self, windows: Vec<GrayImage>) -> Vec<u32>;
}

impl Predictable for Model<'_> {
    fn predict(&mut self, windows: Vec<GrayImage>) -> Vec<u32> {
        let (x, num_image): (Vec<f32>, usize) = (
            windows.iter().flat_map(|img| preprocess(&img)).collect(),
            windows.len(),
        );
        let as_arr = NdArray::from_shape_vec;
        let x = as_arr(ndarray::IxDyn(&[num_image, 28 * 28]), x).unwrap();

        let prediction = self.env.run(|ctx| {
            let logits = compute_logits(ctx, false);
            let predictions = T::argmax(logits, -1, true);
            ctx.evaluator().push(predictions).feed("x", x.view()).run()[0]
                .as_ref()
                .unwrap()
                .clone()
        });
        prediction.iter().map(|v| *v as u32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssd::dataset::DataSet;
    use crate::ssd::trainable::Trainable;
    use image::open;
    use mosse::utils::window_crop;

    #[test]
    fn test_training() {
        let mut dataset = DataSet::new("res/training/".to_string(), "res/labels.txt".to_string());
        dataset.load(false);
        assert_eq!(dataset.samples(), 8);
        let mut model = Model::new();
        model.train(&dataset, 10);
    }

    #[test]
    fn test_predict() {
        let webcam1 = open("res/webcam01.jpg").unwrap().to_luma8();
        let loco5 = window_crop(&webcam1, 28, 28, (300, 400));
        let marker1 = window_crop(&webcam1, 28, 28, (600, 100));
        let marker2 = window_crop(&webcam1, 28, 28, (100, 50));

        let images = vec![loco5, marker1, marker2];

        let mut dataset = DataSet::new("res/training/".to_string(), "res/labels.txt".to_string());
        dataset.load(true);
        assert_eq!(dataset.samples(), 168);
        let mut model = Model::new();
        model.train(&dataset, 100);
        assert_eq!(model.predict(images), vec![5, 1, 2]);
    }
}
