//! An example of template matching in a greyscale image.

use image::{open, DynamicImage, GenericImage, GrayImage, Luma, RgbImage};
use imageproc::definitions::Image;
use imageproc::map::map_colors;
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};
use std::env;
use std::f32;
use std::fs;
use std::path::PathBuf;

struct TemplateMatchingArgs {
    template_path: PathBuf,
    input_path: PathBuf,
    output_dir: PathBuf,
    template_w: u32,
    template_h: u32,
}

impl TemplateMatchingArgs {
    fn parse(args: Vec<String>) -> TemplateMatchingArgs {
        if args.len() != 6 {
            panic!(
                r#"
Usage:

     cargo run --example template_matching template_path input_path output_dir template_w template_h

Loads the image at input_path and extracts a region with the given location and size to use as the matching
template. Calls match_template on the input image and this template, and saves the results to output_dir.
"#
            );
        }
        let template_path = PathBuf::from(&args[1]);
        let input_path = PathBuf::from(&args[2]);
        let output_dir = PathBuf::from(&args[3]);
        let template_w = args[4].parse().unwrap();
        let template_h = args[5].parse().unwrap();

        TemplateMatchingArgs {
            template_path,
            input_path,
            output_dir,
            template_w,
            template_h,
        }
    }
}

/// Convert an f32-valued image to a 8 bit depth, covering the whole
/// available intensity range.
fn convert_to_gray_image(image: &Image<Luma<f32>>) -> GrayImage {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    for p in image.iter() {
        lo = if *p < lo { *p } else { lo };
        hi = if *p > hi { *p } else { hi };
    }

    let range = hi - lo;
    let scale = |x| (255.0 * (x - lo) / range) as u8;
    map_colors(image, |p| Luma([scale(p[0])]))
}

fn run_match_template(
    args: &TemplateMatchingArgs,
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> RgbImage {
    // Match the template and convert to u8 depth to display
    let result = match_template(&image, &template, method);
    println!("extremes {:?}", find_extremes(&result));

    let result_scaled = convert_to_gray_image(&result);

    // Pad the result to the same size as the input image, to make them easier to compare
    let mut result_padded = GrayImage::new(image.width(), image.height());
    result_padded
        .copy_from(&result_scaled, args.template_w / 2, args.template_h / 2)
        .unwrap();
    DynamicImage::ImageLuma8(result_padded).to_rgb8()
}

fn main() {
    let args = TemplateMatchingArgs::parse(env::args().collect());

    let template_path = &args.template_path;
    let input_path = &args.input_path;
    let output_dir = &args.output_dir;

    if !output_dir.is_dir() {
        fs::create_dir(output_dir).expect("Failed to create output directory")
    }

    if !input_path.is_file() {
        panic!("Input file does not exist");
    }

    // Load image and convert to grayscale
    let image = open(input_path)
        .expect(&format!("Could not load image at {:?}", input_path))
        .to_luma8();

    let template = open(template_path)
        .expect(&format!("Could not load image at {:?}", template_path))
        .thumbnail(args.template_w, args.template_h)
        .to_luma8();

    // Match using all available match methods
    let sse = run_match_template(
        &args,
        &image,
        &template,
        MatchTemplateMethod::SumOfSquaredErrors,
    );
    let sse_norm = run_match_template(
        &args,
        &image,
        &template,
        MatchTemplateMethod::SumOfSquaredErrorsNormalized,
    );

    println!(
        "extremes sse {:?}",
        find_extremes(&DynamicImage::ImageRgb8(sse.clone()).to_luma8())
    );
    println!(
        "extremes sse norm {:?}",
        find_extremes(&DynamicImage::ImageRgb8(sse_norm.clone()).to_luma8())
    );

    // Save images to output_dir
    let template_path = output_dir.join("template.png");
    template.save(&template_path).unwrap();
    let source_path = output_dir.join("image.png");
    image.save(&source_path).unwrap();
    let sse_path = output_dir.join("result_sse.png");
    sse.save(&sse_path).unwrap();
    let sse_path = output_dir.join("result_sse_norm.png");
    sse_norm.save(&sse_path).unwrap();
}
