use yew::prelude::*;

pub mod header;
pub mod webcam;

#[function_component(App)]
pub fn app() -> Html {
    html! {
        <>
        <header::Header />
        <webcam::WebCam width=640 height=480 />
        </>
    }
}
