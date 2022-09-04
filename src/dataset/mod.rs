use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, warp, Interpolation, Projection};

#[cfg(not(target_arch = "wasm32"))]
mod folder_dataset;

pub fn window_crop(input_frame: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> RgbImage {
    imageops::crop(&mut input_frame.clone(), x, y, width, height).to_image()
}

pub fn rotated_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    [
        0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5, 2.0,
        -2.0,
    ]
    .iter()
    .map(|rad| rotate_about_center(frame, *rad, Interpolation::Nearest, Rgb([0, 0, 0])))
}

pub fn scaled_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    [0.8, 0.9, 1.1, 1.2].into_iter().map(|scalefactor| {
        let scale = Projection::scale(scalefactor, scalefactor);

        warp(frame, &scale, Interpolation::Nearest, Rgb([0, 0, 0]))
    })
}

pub trait DataSet {
    fn load(&mut self, augment: bool);
    fn generate_random_annotations(&mut self, count_each: usize);
    fn samples(&self) -> usize;
    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>);
}

#[cfg(not(target_arch = "wasm32"))]
pub use folder_dataset::FolderDataSet;
