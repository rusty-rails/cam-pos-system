use super::lenet::Lenet;
use super::{dataset::NdArray, preprocess};
use crate::bbox::BBox;
use crate::detection::{merge, nms_sort, Detection};
use async_trait::async_trait;
use autograph::device::{buffer::*, Device};
use autograph::float_tensor;
use autograph::float_tensor::FloatTensor4;
use autograph::learn::Infer;
use autograph::scalar::Scalar;
use autograph::tensor::{Tensor, TensorView};
use autograph::{
    learn::neural_network::layer::{Conv, Dense, Forward, Layer, MaxPool, Relu},
    result::Result,
};
use autograph::buffer::*;
use autograph::{
    device::{shader::Module},
    buffer::{Buffer, Slice},
};
use image::{DynamicImage, RgbImage, Rgba, SubImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ndarray::{Array1, Array4, Axis};
use rusttype::{Font, Scale};

#[async_trait]
pub trait Predictable {
    async fn predict(&self, windows: Vec<RgbImage>) -> Vec<u32>;
    async fn predict_image(&self, windows: RgbImage) -> Vec<u32>;
    async fn predict_to_image(&self, image: RgbImage) -> DynamicImage;
}

#[async_trait]
impl Predictable for Lenet {
    async fn predict(&self, windows: Vec<RgbImage>) -> Vec<u32> {
        let (x, num_image): (Vec<f32>, usize) = (
            windows.iter().flat_map(|img| preprocess(&img)).collect(),
            windows.len(),
        );
        let as_arr = NdArray::from_shape_vec;
        let x = as_arr(
            ndarray::IxDyn(&[num_image, self.input_width * self.input_height * 3]),
            x,
        )
        .unwrap();
        Vec::new()

        /*let prediction = self.env.run(|ctx| {
            let logits = compute_logits(
                ctx,
                self.input_width as isize,
                self.input_height as isize,
                false,
            );
            let predictions = T::argmax(logits, -1, true);
            ctx.evaluator().push(predictions).feed("x", x.view()).run()[0]
                .as_ref()
                .unwrap()
                .clone()
        });
        prediction.iter().map(|v| *v as u32).collect()
        */
    }

    async fn predict_image(&self, image: RgbImage) -> Vec<u32> {
        let (_cols, _rows, windows) = windows(&image, self.input_width as u32);

        let (x, num_image): (Vec<f32>, usize) = (
            windows
                .iter()
                .flat_map(|img| preprocess(&img.to_image()))
                .collect(),
            windows.len(),
        );

        let device = Device::new().unwrap();

        let x = Array4::from_shape_vec(
            (
                num_image,
                self.input_width as usize,
                self.input_height as usize,
                3 as usize,
            ),
            x,
        )
        .unwrap()
        .into_shape((
            num_image,
            3 as usize,
            self.input_height as usize,
            self.input_width as usize,
        ))
        .unwrap();
        let x = x
            .iter()
            .copied()
            .collect::<Array1<f32>>()
            .into_shape([
                num_image,
                3 as usize,
                self.input_height as usize,
                self.input_width as usize,
            ])
            .unwrap();
        let x = smol::block_on(Tensor::from(x).into_device(device.clone()))
            .unwrap()
            // normalize the bytes to f32
            .scale_into::<f32>(1. / 255.)
            .unwrap()
            .into_float();
        let x = x.into_device(device.clone()).await.unwrap();
        let prediction = self
            .net
            .clone()
            .into_device(device)
            .await
            .unwrap()
            .infer(&x)
            .unwrap();
        let prediction = prediction
            .as_raw_slice()
            .into_device(Device::host())
            .await
            .unwrap();
        let prediction = prediction.into_device(Device::host()).await.unwrap();

       
        println!("{:?}", prediction);
        //prediction.view()

        /*
        let prediction = self.env.run(|ctx| {
            let logits = compute_logits(
                ctx,
                self.input_width as isize,
                self.input_height as isize,
                false,
            );
            let predictions = T::argmax(logits, -1, true);
            ctx.evaluator().push(predictions).feed("x", x.view()).run()[0]
                .as_ref()
                .unwrap()
                .clone()
        });
        */
        //prediction..iter().map(|v| *v as u32).collect()
        //vec![prediction.to_slice().unwrap()]
        todo!()
    }

    async fn predict_to_image(&self, image: RgbImage) -> DynamicImage {
        let window_size = self.input_width as u32;
        let cols = 2 * (image.width() / window_size);
        let rows = 2 * (image.height() / window_size);
        let predictions = self.predict_image(image.clone()).await;
        let detections = detect_objects(cols, rows, predictions, window_size);

        let mut img_copy = DynamicImage::ImageRgb8(image).to_rgba8();
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
                    (detection.bbox.x as u32) as i32,
                    (detection.bbox.y as u32) as i32,
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
                detection.bbox.x as u32,
                detection.bbox.y as u32,
                Scale::uniform(FONT_SCALE),
                &font,
                &format!("{}", detection.class),
            );
        }
        DynamicImage::ImageRgba8(img_copy)
    }
}

pub fn windows(image: &RgbImage, window_size: u32) -> (u32, u32, Vec<SubImage<&RgbImage>>) {
    let cols = 2 * (image.width() / window_size);
    let rows = 2 * (image.height() / window_size);
    let mut subimages = Vec::new();

    for y in 0..rows - 1 {
        for x in 0..cols - 1 {
            subimages.push(SubImage::new(
                image,
                x * (window_size / 2),
                y * (window_size / 2),
                window_size,
                window_size,
            ))
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
    for y in 0..rows - 1 {
        for x in 0..cols - 1 {
            let i = (y * (cols - 1) + x) as usize;
            let class = predictions[i] as usize;
            if class > 0 {
                // add 0.1 to generate an overlap on contacting windows.
                let size = 0.01 + window_size as f64;
                let bbox = BBox {
                    x: (x * (window_size / 2)) as f64,
                    y: (y * (window_size / 2)) as f64,
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
    use super::super::window_crop;
    use super::*;
    use crate::ssd::dataset::DataSet;
    use crate::ssd::trainable::Trainable;
    use image::open;

    const LABELS: usize = 18;
    const IMAGES_PER_LABEL: usize = 21;

    #[test]
    fn test_training() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), 18);
        let mut model = Lenet::new(28, 28);
        model.train(&dataset, 10);
    }

    #[tokio::test]
    async fn test_predict() {
        let window_size = 28;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (280, 370));
        let marker1 = window_crop(&webcam1, window_size, window_size, (540, 90));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 25));

        let images = vec![loco5, marker1, marker2];

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
        let mut model = Lenet::new(28, 28);
        model.train(&dataset, 100);
        //assert_eq!(model.predict(images), vec![5, 1, 2]);
        assert_eq!(
            model
                .predict(images.clone())
                .await
                .first()
                .as_ref()
                .unwrap(),
            &&5
        );
        assert_eq!(model.predict(images).await.last().unwrap(), &2);
    }

    #[test]
    fn test_windows_cols_and_rows() {
        let image = RgbImage::new(10, 10);
        let window_size = 4;
        let (cols, rows, subimages) = windows(&image, window_size);

        assert_eq!(cols, 4);
        assert_eq!(rows, 4);
        assert_eq!(subimages.len() as u32, (cols - 1) * (rows - 1));
    }

    #[tokio::test]
    async fn test_predict_image() {
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
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
        let mut model = Lenet::new(28, 28);
        model.train(&dataset, 25);
        assert!(model.predict_image(webcam1).await.len() > 0);
    }

    #[test]
    fn test_detect_objects() {
        let (cols, rows) = (4, 4);
        let window_size = 4;
        let predictions = vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0];

        let detections = detect_objects(cols, rows, predictions, window_size);
        assert_eq!(detections.len(), 1);
    }

    #[tokio::test]
    async fn test_predict_to_image() {
        let window_size = 64;
        let webcam1 = open("res/webcam01.jpg").unwrap().to_rgb8();
        let loco5 = window_crop(&webcam1, window_size, window_size, (280, 370));
        let marker1 = window_crop(&webcam1, window_size, window_size, (540, 90));
        let marker2 = window_crop(&webcam1, window_size, window_size, (100, 25));

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        dataset.generate_random_annotations(25);
        println!(
            "{:?}",
            dataset
                .data
                .iter()
                .map(|(l, _)| l.to_string())
                .collect::<Vec<String>>()
        );
        let mut model = Lenet::new(28, 28);
        model.train(&dataset, 25).await.unwrap();

        model
            .predict_to_image(webcam1)
            .await
            .save("out/test_predict_to_image_webcam1.png")
            .unwrap();
        model
            .predict_to_image(loco5)
            .await
            .save("out/test_predict_to_image_loco5.png")
            .unwrap();
        model
            .predict_to_image(marker1)
            .await
            .save("out/test_predict_to_image_marker1.png")
            .unwrap();
        model
            .predict_to_image(marker2)
            .await
            .save("out/test_predict_to_image_marker2.png")
            .unwrap();
    }
}
