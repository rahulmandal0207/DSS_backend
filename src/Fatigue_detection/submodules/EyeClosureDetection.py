import cv2 as cv
from mediapipe.python import solutions
import numpy as np

class EyeClosureDetection:
    def __init__(self, eye_closure_threshold=0.25):

        self.EYE_CLOSURE_THRESHOLD = eye_closure_threshold

    def __eye_aspect_ratio(self, eye):

        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))

        ear = (A + B) / (2.0 * C)
        return ear

    def draw_landmark(self, frame, landmarks):
        both_eye_points = [33,160,158,133,153,144,362,385,387,263,373,380]
        for idx in both_eye_points:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            # cv.putText(frame, str(idx), (x,y), cv.FONT_HERSHEY_SIMPLEX, .3,(0,255,0),1)
            cv.circle(frame,(x,y),2,(0,0,255),-1)

    def detect_eye_closure(self, face_landmarks):

        left_eye = [
            face_landmarks.landmark[33],  # Left eye corner
            face_landmarks.landmark[160],  # Top eyelid
            face_landmarks.landmark[158],  # Bottom eyelid
            face_landmarks.landmark[133],  # Right eye corner
            face_landmarks.landmark[153],  # Top eyelid
            face_landmarks.landmark[144]  # Bottom eyelid
        ]

        right_eye = [
            face_landmarks.landmark[362],  # Right eye corner
            face_landmarks.landmark[385],  # Top eyelid
            face_landmarks.landmark[387],  # Bottom eyelid
            face_landmarks.landmark[263],  # Left eye corner
            face_landmarks.landmark[373],  # Top eyelid
            face_landmarks.landmark[380]  # Bottom eyelid
        ]

        left_eye_ear = self.__eye_aspect_ratio(left_eye)
        right_eye_ear = self.__eye_aspect_ratio(right_eye)

        left_eye_closed = left_eye_ear < self.EYE_CLOSURE_THRESHOLD
        right_eye_closed = right_eye_ear < self.EYE_CLOSURE_THRESHOLD

        if left_eye_closed or right_eye_closed:
            return True, face_landmarks

        return False, None

    def process_frame(self, frame, face_landmarks):
        eye_closure, landmarks = self.detect_eye_closure(face_landmarks)

        if eye_closure:
            cv.putText(frame, "Eye Closure Detected!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.draw_landmark(frame, landmarks)
        else:
            cv.putText(frame, "No Eye Closure", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame


if __name__ == "__main__":

    mp_face_mesh = solutions.face_mesh
    mp_face = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    ed = EyeClosureDetection()

    cap = cv.VideoCapture(0)

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("No frame found")
            exit()

        frame = cv.flip(frame,1)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = mp_face.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                output_frame = ed.process_frame(frame, face_landmarks)

        cv.imshow("Frame", output_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
