import cv2 as cv
from mediapipe.python import solutions
import numpy as np

class HeadTiltDetection:
    def __init__(self, tilt_threshold=15):
        self.TILT_THRESHOLD = tilt_threshold  # degrees

    def __calculate_angle(self, left_eye_point, right_eye_point):
        x1, y1 = left_eye_point
        x2, y2 = right_eye_point
        radians = np.arctan2(y2 - y1, x2 - x1)
        angle = np.degrees(radians)
        return angle

    def detect_tilt(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        left_eye_outer = face_landmarks.landmark[33]   # Left eye corner
        right_eye_outer = face_landmarks.landmark[263] # Right eye corner

        left = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
        right = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))

        angle = self.__calculate_angle(left, right)

        is_tilted = abs(angle) > self.TILT_THRESHOLD


        return is_tilted, face_landmarks, angle, [left, right]

    def draw_landmarks(self, frame, is_tilted, angle, points):
        color = (0, 0, 255) if is_tilted else (0, 255, 0)
        for pt in points:
            cv.circle(frame, pt, 3, color, -1)

        text = f"Head Tilt: {int(angle)}"
        cv.putText(frame, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def process_frame(self, frame, face_landmarks):
        is_tilted, landmarks, angle, points = self.detect_tilt(face_landmarks, frame.shape)

        if is_tilted:
            cv.putText(frame, "Head Tilt Detected!", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv.putText(frame, "Head Straight", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.draw_landmarks(frame, is_tilted, angle, points)



        return frame, angle


if __name__ == "__main__":
    mp_face_mesh = solutions.face_mesh
    mp_face = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    htd = HeadTiltDetection()
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frame found")
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = mp_face.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame, angle = htd.process_frame(frame, face_landmarks)

        cv.imshow("Head Tilt Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
