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

setTimeout(pause_webcam, 500);