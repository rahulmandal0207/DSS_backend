import time
import cv2 as cv
import mediapipe.python.solutions as mp
from flask import Flask, Response, render_template
from flask_cors import CORS
from threading import Thread, Lock
from queue import Queue

from src.Fatigue_detection.submodules.EyeClosureDetection import EyeClosureDetection
from src.Fatigue_detection.submodules.YawnDetection import YawnDetection

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.drawing_utils
mp_drawing_style = mp.drawing_styles

# Initialize detection modules
ec = EyeClosureDetection()
yd = YawnDetection()

# Initialize video capture
cap = cv.VideoCapture(0)

# Thread-safe queue to share frames
frame_queue = Queue(maxsize=10)
lock = Lock()

def video_capture_thread():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read the frame")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Add the frame and landmarks to the queue
                with lock:
                    if not frame_queue.full():
                        frame_queue.put((frame.copy(), face_landmarks))
                    else:
                        # Drop the oldest frame if the queue is full
                        frame_queue.get()
                        frame_queue.put((frame.copy(), face_landmarks))

    cap.release()

# Start the video capture thread
Thread(target=video_capture_thread, daemon=True).start()

def generate_frames(processing_function):
    while True:
        with lock:
            if not frame_queue.empty():
                frame, face_landmarks = frame_queue.get()
                processed_frame = processing_function(frame, face_landmarks)
            else:
                continue

        # Encode the frame as JPEG
        _, buffer = cv.imencode(".jpg", processed_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame for streaming
        yield (
            b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.route("/video_feed/yawn")
def video_feed_yawn():
    return Response(
        generate_frames(yd.process_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_feed/eye_closure")
def video_feed_eye_closure():
    return Response(
        generate_frames(ec.process_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# @app.route("/video_feed/original")
# def video_feed_original():
#     return Response(
#         generate_frames(original_frame),
#         mimetype="multipart/x-mixed-replace; boundary=frame"
#     )

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)