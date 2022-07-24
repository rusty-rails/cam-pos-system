var selectMode = "";
var webcam = document.getElementById("webcam");

function play_webcam() {
  webcam.setAttribute("src", "/tracked-stream");
  webcam.onclick = function () {
    pause_webcam();
  };
}

function pause_webcam() {
  webcam.setAttribute("src", "/tracked");
  webcam.onclick = function () {
    play_webcam();
  };
}

function setSelectMode(mode) {
  selectMode = mode;
  console.log(mode);
}

async function updatePosition(event) {
  let x = event.offsetX;
  let y = event.offsetY;
  let id = 0;
  if (selectMode == "train") {
    id = 5;
  } else if (selectMode == "marker1") {
    id = 1;
  } else if (selectMode == "marker2") {
    id = 2;
  } else if (selectMode == "marker3") {
    id = 3;
  } else if (selectMode == "marker4") {
    id = 4;
  }

  const data = {
    x,
    y,
    id,
  };

  const response = await fetch("/update-position", {
    method: "POST",
    mode: "cors",
    cache: "no-cache",
    headers: {
      "Content-Type": "application/json",
    },
    redirect: "follow",
    referrerPolicy: "no-referrer",
    body: JSON.stringify(data),
  }).catch(console.log);
}
