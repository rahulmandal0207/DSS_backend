import time

import cv2 as cv
from flask import Flask, Response, render_template
from flask_cors import CORS

from src.Fatigue_detection.FatigueDetection import FatigueDetection

app = Flask(__name__)
CORS(app)

fd = FatigueDetection()

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("resources/SampleVideo_640x360_5mb.mp4")

def generate_frames(processing_function):

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.416

    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()

        if not success:
            print("Failed to read the frame")
            break;

        frame = cv.flip(frame, 1)

        processed_frame = processing_function(frame)

        _,buffer = cv.imencode(".jpg", processed_frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

        elapsed_time = time.time() - start_time
        sleep_time = max(frame_time - elapsed_time, 0)
        time.sleep(sleep_time)

    cap.release()

def original_frame(frame):
    return frame

@app.route("/video_feed/yawn")
def video_feed_yawn():
    return  Response(
        generate_frames(fd.process_frame),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_feed/original")
def video_feed_original():
    return Response(
        generate_frames(original_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_feed/gray")
def video_feed_gray():
    return Response(
        # generate_frames(test.grayscale_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)








