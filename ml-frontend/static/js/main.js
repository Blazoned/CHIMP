const VID_WIDTH = 1280, VID_HEIGHT = 720;
const HIDDEN_CANVAS_WIDTH = 320, HIDDEN_CANVAS_HEIGHT = 180;
// const HIDDEN_CANVAS_WIDTH = 640, HIDDEN_CANVAS_HEIGHT = 360;

let sock;
let video_origin, canvas_origin;

let has_recently_updated_data = false;
let PREDICTION_TIMEOUT = 500;

function init() {
    // MEDIA WEBCAM CAPTURE
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        alert("Your browser doesn't seem to support the use of a webcam. Please use a more modern browser.");
        return;
    }

    video_origin = document.createElement('video');
    video_origin.id = 'video_origin';
    video_origin.width = VID_WIDTH;
    video_origin.height = VID_HEIGHT;

    canvas_origin = document.createElement('canvas');
    canvas_origin.width = HIDDEN_CANVAS_WIDTH;
    canvas_origin.height = HIDDEN_CANVAS_HEIGHT;

    navigator.mediaDevices.getUserMedia({
            video: true
        })
        .then(stream => {
            video_origin.srcObject = stream;
            video_origin.onloadedmetadata = (e) => video_origin.play();
        })
        .catch(msg => console.log('Error: ' + msg));


    // SOCKET.IO
    sock = io.connect('http://' + document.domain + ':' + location.port);

    sock.on('connect',
        function() {
            console.log('Initialised SocketIO connection...');

            // START CAPTURE
            capture();
        });

    sock.on('disconnect',
        function() {
            console.log('Terminated SocketIO connection.');
        });

    sock.on('update-data', (data) =>
        {
            console.log(data);

            if (!has_recently_updated_data) {
                has_recently_updated_data = true;

                data.forEach(face => {
                    face.forEach(emotion => {
                        prediction_text = (emotion[1]*100).toFixed(2) + '%';

                        switch (emotion[0]) {
                            case 'angry':
                                data_angry.innerText = prediction_text;
                                break;
                            case 'disgust':
                                data_disgust.innerText = prediction_text;
                                break;
                            case 'fear':
                                data_fear.innerText = prediction_text;
                                break;
                            case 'happy':
                                data_happy.innerText = prediction_text;
                                break;
                            case 'neutral':
                                data_neutral.innerText = prediction_text;
                                break;
                            case 'sad':
                                data_sad.innerText = prediction_text;
                                break;
                            case 'surprise':
                                data_surprise.innerText = prediction_text;
                                break;
                        }
                    });
                });

                setTimeout(function () {
                    has_recently_updated_data = false;
                }, PREDICTION_TIMEOUT)
            }
        });
}

// CAPTURE AND MANIPULATE WEBCAM FEED
const capture = () => {
    canvas_origin.getContext('2d').drawImage(video_origin, 0, 0, canvas_origin.width, canvas_origin.height);
    canvas_origin.toBlob((blob) => {
        sock.emit('process-image', blob, (data) => {
            let imgData = new Blob([data], {type: 'image/jpg'})
            let img = new Image()
            img.onload = () => preview.getContext('2d').drawImage(img, 0, 0, preview.width, preview.height);
            img.src = URL.createObjectURL(imgData)
            capture()
        });
    });
}
