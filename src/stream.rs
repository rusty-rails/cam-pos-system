use crate::frame::Frame;
use format_bytes::format_bytes;
use futures::Stream;
use image::DynamicImage;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Context;
use std::task::Poll;
use std::thread;
use tokio::time;

const FRAME_MILLIS: u32 = 1000 / 2;

pub struct MJpeg {
    pub frame: Arc<Mutex<Frame>>,
}

impl MJpeg {
    pub fn new(frame: Arc<Mutex<Frame>>) -> Self {
        Self { frame }
    }
}

impl Stream for MJpeg {
    type Item = Vec<u8>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let start = time::Instant::now();

        let buf: Vec<u8> = {
            let frame = self.frame.lock().unwrap();
            let base_img = frame.tracked.clone();
            let base_img: DynamicImage = DynamicImage::ImageRgb8(base_img);
            let mut buf = vec![];
            base_img
                .write_to(&mut buf, image::ImageOutputFormat::Jpeg(70))
                .unwrap();
            buf
        };
        let data = format_bytes!(b"\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n{}", &buf);
        let duration = time::Instant::now() - start;
        thread::sleep(time::Duration::from_millis(
            (FRAME_MILLIS as i32 - duration.as_millis() as i32).max(0) as u64,
        ));
        Poll::Ready(Some(data))
    }
}
