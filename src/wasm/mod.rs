use crate::tracker::Tracker;
use wasm_bindgen::prelude::*;
use web_sys;

pub mod gui;

#[wasm_bindgen]
pub struct TrackerJS {
    tracker: Tracker,
}

#[wasm_bindgen]
impl TrackerJS {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> TrackerJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();

        let tracker = Tracker::new(width, height);
        TrackerJS { tracker }
    }

    #[wasm_bindgen]
    pub fn set_target(&mut self, x: u32, y: u32, img_data: &[u8]) {
        let img = image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();
        self.tracker
            .tracker
            .add_or_replace_target(1, (x, y), &img.to_luma8());
    }

    #[wasm_bindgen]
    pub fn next(&mut self, img_data: &[u8]) -> Vec<u8> {
        let mut img =
            image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();
        img = self.tracker.next(&img.to_luma8());

        let mut image_data: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut image_data),
            image::ImageFormat::Png,
        )
        .unwrap();
        image_data
    }
}

#[wasm_bindgen]
pub fn main(root: web_sys::Element) {
    wasm_logger::init(wasm_logger::Config::default());
    yew::Renderer::<gui::App>::with_root(root).render();
}
