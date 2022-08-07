// source: https://github.com/security-union/yew-beyond-hello-world/blob/main/src/main.rs
// MIT License
// Copyright (c) 2022 Security Union

use crate::tracker::Tracker;
use image::{DynamicImage, ImageBuffer};
use js_sys::*;
use std::sync::{Arc, Mutex};
use wasm_bindgen::{Clamped, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::*;
use yew::{html, Component, Context, Html, Properties};

pub enum Msg {
    Clicked(i32, i32),
    Frame,
}

#[derive(PartialEq, Properties)]
pub struct Props {
    pub width: u32,
    pub height: u32,
}

pub struct WebCam {
    width: u32,
    height: u32,
    tracker: Arc<Mutex<Tracker>>,
    //reader: web_sys::ReadableStreamDefaultReader,
    canvas: Arc<Mutex<HtmlCanvasElement>>,
    frame: Arc<Mutex<Option<DynamicImage>>>,
}

impl WebCam {
    fn set_target(&mut self, x: u32, y: u32) {
        let frame = self.frame.lock().unwrap();
        match frame.as_ref() {
            Some(img) => {
                let mut tracker = self.tracker.lock().unwrap();
                tracker.tracker.add_target(1, (x, y), &img.to_luma8())
            }
            None => (),
        }
    }

    pub fn set_frame(&mut self, img_data: &[u8]) {
        let img = image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();
        let canvas = self.canvas.lock().unwrap();
        let context = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .unwrap();
        let img_rgba = img.to_rgba8();
        let clamped_buf: Clamped<&[u8]> = Clamped(img_rgba.as_raw());
        let image_data_temp =
            ImageData::new_with_u8_clamped_array_and_sh(clamped_buf, self.width, self.height)
                .unwrap();
        context.put_image_data(&image_data_temp, 0.0, 0.0).unwrap();
        self.frame = Arc::new(Mutex::new(Some(img)));
    }

    pub async fn get_reader(width: u32, height: u32) -> web_sys::ReadableStreamDefaultReader {
        let navigator = window().unwrap().navigator();
        let media_devices = navigator.media_devices().unwrap();
        let video_element = window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("webcam")
            .unwrap()
            .unchecked_into::<HtmlVideoElement>();

        let mut constraints = MediaStreamConstraints::new();
        constraints.video(&Boolean::from(true));
        constraints.audio(&Boolean::from(false));
        let devices_query = media_devices
            .get_user_media_with_constraints(&constraints)
            .unwrap();
        let device = JsFuture::from(devices_query)
            .await
            .unwrap()
            .unchecked_into::<MediaStream>();
        video_element.set_src_object(Some(&device));
        let video_track = Box::new(
            device
                .get_video_tracks()
                .find(&mut |_: JsValue, _: u32, _: Array| true)
                .unchecked_into::<VideoTrack>(),
        );
        let settings = &mut video_track
            .clone()
            .unchecked_into::<MediaStreamTrack>()
            .get_settings();
        settings.width(width as i32);
        settings.height(height as i32);
        let processor = MediaStreamTrackProcessor::new(&MediaStreamTrackProcessorInit::new(
            &video_track.unchecked_into::<MediaStreamTrack>(),
        ))
        .unwrap();
        let reader = processor
            .readable()
            .get_reader()
            .unchecked_into::<ReadableStreamDefaultReader>();
        reader
    }

    pub fn start(&mut self) {
        let width = self.width;
        let height = self.height;
        let frame = self.frame.clone();
        let canvas = self.canvas.clone();
        let tracker = self.tracker.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let reader = Self::get_reader(width, height).await;
            loop {
                let result = JsFuture::from(reader.read()).await.map_err(|e| {
                    console::log_1(&e);
                });
                match result {
                    Ok(js_frame) => {
                        let video_frame = Reflect::get(&js_frame, &JsString::from("value"))
                            .unwrap()
                            .unchecked_into::<VideoFrame>();
                        let mut video_vector = vec![0u8; (width * height * 3) as usize];
                        let video_message = video_vector.as_mut();
                        let _ = video_frame.copy_to_with_u8_array(video_message);
                        match ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
                            width,
                            height,
                            video_message.to_vec(),
                        ) {
                            Some(img) => {
                                let canvas = canvas.lock().unwrap();
                                let context = canvas
                                    .get_context("2d")
                                    .unwrap()
                                    .unwrap()
                                    .dyn_into::<web_sys::CanvasRenderingContext2d>()
                                    .unwrap();

                                let mut frame = frame.lock().unwrap();
                                let mut tracker = tracker.lock().unwrap();
                                let img =
                                    tracker.next(&image::DynamicImage::ImageRgb8(img).to_luma8());
                                let img_rgba = img.clone().to_rgba8();
                                let clamped_buf: Clamped<&[u8]> = Clamped(img_rgba.as_raw());
                                let image_data_temp = ImageData::new_with_u8_clamped_array_and_sh(
                                    clamped_buf,
                                    width,
                                    height,
                                )
                                .unwrap();
                                context.put_image_data(&image_data_temp, 0.0, 0.0).unwrap();

                                *frame = Some(img);
                            }
                            None => (),
                        };
                        video_frame.close();
                    }
                    Err(_e) => {
                        console::log_1(&JsString::from("error"));
                    }
                }
            }
        })
    }
}

impl Component for WebCam {
    type Message = Msg;
    type Properties = Props;

    fn create(ctx: &Context<Self>) -> Self {
        let width = ctx.props().width;
        let height = ctx.props().height;
        let tracker = Tracker::new(width, height);
        let window = web_sys::window().expect("should have a window in this context");
        let document = window.document().expect("window should have a document");
        let canvas = document.create_element("canvas").unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| ())
            .unwrap();
        canvas.set_width(width);
        canvas.set_height(height);
        let context = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .unwrap();
        context.set_fill_style(&JsValue::from_str("gray"));
        context.fill_rect(0.0, 0.0, width as f64, height as f64);
        let mut webcam = Self {
            width,
            height,
            tracker: Arc::new(Mutex::new(tracker)),
            canvas: Arc::new(Mutex::new(canvas)),
            frame: Arc::new(Mutex::new(None)),
        };
        webcam.start();
        webcam
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Clicked(x, y) => {
                log::info!("{} {}", x, y);
                self.set_target(x.try_into().unwrap(), y.try_into().unwrap());
            }
            _ => (),
        };
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let node: Node = {
            let canvas = self.canvas.lock().unwrap();
            canvas.clone().into()
        };
        let onclick = ctx.link().callback(|event: MouseEvent| {
            let x = event.offset_x();
            let y = event.offset_y();
            Msg::Clicked(x, y)
        });
        html! {
            <>
            <div hidden=true>
                <video autoplay=true id="webcam"></video>
            </div>
            <div {onclick}>
            {Html::VRef(node)}
            </div>
            </>
        }
    }
}
