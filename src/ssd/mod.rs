use ag::ndarray;
use autograd as ag;
use image::{imageops, RgbImage, Rgb};
use imageproc::geometric_transformations::{rotate_about_center, warp, Interpolation, Projection};
use std::f32;

pub mod dataset;
pub mod model;
pub mod predictable;
pub mod trainable;
pub mod yolo;

pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

pub fn window_crop(
    input_frame: &RgbImage,
    window_width: u32,
    window_height: u32,
    center: (u32, u32),
) -> RgbImage {
    let window = imageops::crop(
        &mut input_frame.clone(),
        center
            .0
            .saturating_sub(window_width / 2)
            .min(input_frame.width() - window_width),
        center
            .1
            .saturating_sub(window_height / 2)
            .min(input_frame.height() - window_height),
        window_width,
        window_height,
    )
    .to_image();

    return window;
}

pub fn rotated_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    // build an iterator that produces training frames that have been slightly rotated according to a theta value.
    let rotated_frames = [
        0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5, 2.0,
        -2.0,
    ]
    .iter()
    .map(|rad| {
        // Rotate an image clockwise about its center by theta radians.
        let training_frame = rotate_about_center(frame, *rad, Interpolation::Nearest, Rgb([0,0,0]));

        #[cfg(debug_assertions)]
        {
            training_frame
                .save(format!("training_frame_rotated_theta_{}.png", rad))
                .unwrap();
        }

        return training_frame;
    });
    rotated_frames
}

pub fn scaled_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    // build an iterator that produces training frames that have been slightly scaled to various degrees ('zoomed')
    let scaled_frames = [0.8, 0.9, 1.1, 1.2].into_iter().map(|scalefactor| {
        let scale = Projection::scale(scalefactor, scalefactor);

        let scaled_training_frame = warp(frame, &scale, Interpolation::Nearest, Rgb([0,0,0]));

        #[cfg(debug_assertions)]
        {
            scaled_training_frame
                .save(format!("training_frame_scaled_{}.png", scalefactor))
                .unwrap();
        }

        return scaled_training_frame;
    });
    scaled_frames
}

pub fn preprocess(image: &RgbImage) -> Vec<f32> {
    let mut prepped: Vec<f32> = image
        .pixels()
        // convert the pixel to u8 and then to f32
        .map(|p| p[0] as f32)
        // add 1, and take the natural logarithm
        .map(|p| (p + 1.0).ln())
        .collect();

    // normalize to mean = 0 (subtract image-wide mean from each pixel)
    let sum: f32 = prepped.iter().sum();
    let mean: f32 = sum / prepped.len() as f32;
    prepped.iter_mut().for_each(|p| *p = *p - mean);

    // normalize to norm = 1, if possible
    let u: f32 = prepped.iter().map(|a| a * a).sum();
    let norm = u.sqrt();
    if norm != 0.0 {
        prepped.iter_mut().for_each(|e| *e = *e / norm)
    }

    // multiply each pixel by a cosine window
    let (width, height) = image.dimensions();
    let mut position = 0;
    for i in 0..width {
        for j in 0..height {
            let cww = ((f32::consts::PI * i as f32) / (width - 1) as f32).sin();
            let cwh = ((f32::consts::PI * j as f32) / (height - 1) as f32).sin();
            prepped[position] = cww.min(cwh) * prepped[position];
            position += 1;
        }
    }

    return prepped;
}