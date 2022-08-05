use super::model::Model;
use super::trainable::compute_logits;
use super::NdArray;
use crate::bbox::BBox;
use crate::detection::{merge, nms_sort, Detection};
use ag::ndarray;
use ag::tensor_ops as T;
use autograd as ag;
use image::{DynamicImage, GrayImage, Rgba, SubImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use mosse::utils::preprocess;
use rusttype::{Font, Scale};

pub trait Predictable {
    fn predict(&mut self, windows: Vec<GrayImage>) -> Vec<u32>;
    fn predict_image(&mut self, windows: GrayImage) -> Vec<u32>;
    fn predict_to_image(&mut self, image: GrayImage) -> DynamicImage;
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

    fn predict_to_image(&mut self, image: GrayImage) -> DynamicImage {
        let window_size = 28;
        let cols = image.width() / window_size;
        let rows = image.height() / window_size;
        let predictions = self.predict_image(image.clone());
        let detections = detect_objects(cols, rows, predictions, window_size);

        let mut img_copy = DynamicImage::ImageLuma8(image).to_rgba8();
        for detection in detections.iter() {
            let color = Rgba([125u8, 255u8, 0u8, 0u8]);
            draw_cross_mut(
                &mut img_copy,
                Rgba([255u8, 0u8, 0u8, 0u8]),
                detection.bbox.x as i32,
                detection.bbox.y as i32,
            );
            draw_hollow_rect_mut(
                &mut img_copy,
                Rect::at(
                    (detection.bbox.x as u32).saturating_sub(window_size / 2) as i32,
                    (detection.bbox.y as u32).saturating_sub(window_size / 2) as i32,
                )
                .of_size(window_size, window_size),
                color,
            );

            let font_data = include_bytes!("../../res/Arial.ttf");
            let font = Font::try_from_bytes(font_data as &[u8]).unwrap();

            const FONT_SCALE: f32 = 10.0;

            draw_text_mut(
                &mut img_copy,
                Rgba([125u8, 255u8, 0u8, 0u8]),
                detection.bbox.x as u32 - (window_size / 2),
                detection.bbox.y as u32 - (window_size / 2),
                Scale::uniform(FONT_SCALE),
                &font,
                &format!("{}", detection.class),
            );
        }
        DynamicImage::ImageRgba8(img_copy)
    }
}

pub fn windows(image: &GrayImage, window_size: u32) -> (u32, u32, Vec<SubImage<&GrayImage>>) {
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

pub fn detect_objects(
    cols: u32,
    rows: u32,
    predictions: Vec<u32>,
    window_size: u32,
) -> Vec<Detection> {
    let mut detections: Vec<Detection> = Vec::new();
    for y in 0..rows {
        for x in 0..cols {
            let i = (y * cols + x) as usize;
            let class = predictions[i] as usize;
            if class > 0 {
                // add 0.1 to generate an overlap on contacting windows.
                let size = 0.01 + window_size as f64;
                let bbox = BBox {
                    x: (x * window_size) as f64,
                    y: (y * window_size) as f64,
                    w: size,
                    h: size,
                };
                detections.push(Detection {
                    class,
                    bbox,
                    confidence: 1.0,
                });
            }
        }
    }
    let detections = merge(detections);
    let detections = nms_sort(detections);
    detections
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

    #[test]
    fn test_detect_objects() {
        let (cols, rows) = (4, 4);
        let window_size = 4;
        let predictions = vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0];

        let detections = detect_objects(cols, rows, predictions, window_size);
        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn test_predict_to_image() {
        let window_size = 64;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_luma8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (300, 400));
        let marker1 = window_crop(&webcam1, window_size, window_size, (600, 100));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 50));

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
        model.train(&dataset, 500);

        model
            .predict_to_image(webcam1)
            .save("out/test_predict_to_image_webcam1.png")
            .unwrap();
        model
            .predict_to_image(loco5)
            .save("out/test_predict_to_image_loco5.png")
            .unwrap();
        model
            .predict_to_image(marker1)
            .save("out/test_predict_to_image_marker1.png")
            .unwrap();
        model
            .predict_to_image(marker2)
            .save("out/test_predict_to_image_marker2.png")
            .unwrap();
    }
}
