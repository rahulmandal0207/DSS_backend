
import cv2 as cv
from mediapipe.python import solutions
from src.Fatigue_detection.submodules.EyeClosureDetection import EyeClosureDetection
from src.Fatigue_detection.submodules.YawnDetection import YawnDetection
import pandas as pd


class FatigueDetection:
    def __init__(self):

        self.yawn_detection = YawnDetection()
        self.eye_closure_detection = EyeClosureDetection()

        self.mp_face_mesh = solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = solutions.drawing_utils
        self.mp_drawing_style = solutions.drawing_styles

    def process_frame(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:

                yawn_frame, mar = self.yawn_detection.process_frame(frame.copy(),face_landmark)

                eye_frame, left_eye_ear, right_eye_ear = self.eye_closure_detection.process_frame(frame.copy(), face_landmark)

                print("MAR: ", mar)
                print("L_EAR: ", left_eye_ear, "R_EAR: ", right_eye_ear)
                # with open('resources/data/output.csv', 'a') as file:
                #     file.write(f"{mar},{left_eye_ear},{right_eye_ear}\n")
                return yawn_frame, eye_frame

        return None, None


if __name__ == '__main__':
    fd = FatigueDetection()

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frames found")
            exit()

        frame = cv.flip(frame,1)

        yawn_frame , eye_closure_frame = fd.process_frame(frame)

        if yawn_frame is not None:
            cv.imshow("Yawn detector", yawn_frame)

        if eye_closure_frame is not None:

            cv.imshow("Eye closure detector", eye_closure_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

