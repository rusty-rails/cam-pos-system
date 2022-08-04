use super::model::Model;
use super::trainable::compute_logits;
use super::NdArray;
use ag::ndarray;
use ag::tensor_ops as T;
use autograd as ag;
use image::{GrayImage, SubImage};
use mosse::utils::preprocess;

pub trait Predictable {
    fn predict(&mut self, windows: Vec<GrayImage>) -> Vec<u32>;
    fn predict_image(&mut self, windows: GrayImage) -> Vec<u32>;
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

    fn predict_image(&mut self, image: GrayImage) -> Vec<u32> {
        let (_cols, _rows, windows) = windows(&image, 28);

        let (x, num_image): (Vec<f32>, usize) = (
            windows
                .iter()
                .flat_map(|img| preprocess(&img.to_image()))
                .collect(),
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

fn windows(image: &GrayImage, window_size: u32) -> (u32, u32, Vec<SubImage<&GrayImage>>) {
    let cols = image.width() / window_size;
    let rows = image.height() / window_size;
    let mut subimages = Vec::new();

    for y in 0..rows {
        for x in 0..cols {
            subimages.push(SubImage::new(image, x, y, window_size, window_size))
        }
    }

    return (cols, rows, subimages);
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

    #[test]
    fn test_windows_cols_and_rows() {
        let image = GrayImage::new(10, 10);
        let window_size = 4;
        let (cols, rows, subimages) = windows(&image, window_size);

        assert_eq!(cols, 2);
        assert_eq!(rows, 2);
        assert_eq!(subimages.len() as u32, cols * rows);
    }

    #[test]
    fn test_predict_image() {
        let webcam1 = open("res/webcam01.jpg").unwrap().to_luma8();

        let mut dataset = DataSet::new("res/training/".to_string(), "res/labels.txt".to_string());
        dataset.load(true);
        dataset.generate_random_annotations(100);
        println!(
            "{:?}",
            dataset
                .data
                .iter()
                .map(|(l, _)| l.to_string())
                .collect::<Vec<String>>()
        );
        let mut model = Model::new();
        model.train(&dataset, 100);
        assert!(model.predict_image(webcam1).len() > 0);
    }
}
