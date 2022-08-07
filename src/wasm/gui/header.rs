use yew::{html, Component, Context, Html};

pub struct Header;

impl Component for Header {
    type Message = ();
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
        html! {
            <div class="jumbotron mt-4 p-3 mb-5 bg-light rounded shadow">
                <h1>{"Camera Positioning System"}</h1>
            </div>
        }
    }
}
