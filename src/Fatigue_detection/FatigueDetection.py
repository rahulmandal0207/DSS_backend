import time

import cv2 as cv
from mediapipe.python import solutions
from scipy.sparse import eye_array
from tensorflow.python.ops.gen_summary_ops import write_histogram_summary_eager_fallback

from src.Fatigue_detection.submodules.EyeClosureDetection import EyeClosureDetection
from src.Fatigue_detection.submodules.HeadTiltDetection import HeadTiltDetection
from src.Fatigue_detection.submodules.YawnDetection import YawnDetection
import  os

from src.Train.make_dataset import make_dataset


class FatigueDetection:
    def __init__(self):

        self.yawn_detection = YawnDetection()
        self.eye_closure_detection = EyeClosureDetection(
            eye_closure_threshold=.2
        )
        self.head_tile_detection = HeadTiltDetection()

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

                head_frame, angle = self.head_tile_detection.process_frame(frame.copy(), face_landmark)

                print("MAR: ", mar,"L_EAR: ", left_eye_ear, "R_EAR: ", right_eye_ear, "H_TILT:", angle)

                # make_dataset('drowsy1',mar,left_eye_ear,right_eye_ear)
                return yawn_frame, eye_frame, head_frame

        return None, None


if __name__ == '__main__':
    fd = FatigueDetection()

    cap = cv.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frame")
            exit()

        y_frame, e_frame, h_frame = fd.process_frame(frame)

        if y_frame is not None:
            cv.imshow("Yawn", y_frame)

        if e_frame is not None:
            cv.imshow("Eye", e_frame)

        if h_frame is not None:
            cv.imshow("head", h_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    # drowsy_path = "../../resources/data/train_data_1/drowsy"
    # drowsy_images = os.listdir(drowsy_path)
    #
    # not_drowsy_path = "../../resources/data/train_data_1/notdrowsy"
    # not_drowsy_images = os.listdir(not_drowsy_path)
    # # print(not_drowsy_images)
    #
    # for image in drowsy_images:
    #     frame = cv.imread(f"{drowsy_path}/{image}")
    #
    #     y_frame, e_frame = fd.process_frame(frame)
    #
    #     cv.imshow("MAR",y_frame)
    #     cv.imshow("EAR", e_frame)
    #
    #     cv.waitKey(1)
    #     cv.destroyAllWindows()

    # for image in not_drowsy_images[0:1000]:
    #     frame = cv.imread(f"{not_drowsy_path}/{image}")
    #
    #     y_frame, e_frame = fd.process_frame(frame)
    #
    #     # cv.imshow("MAR",y_frame)
    #     # cv.imshow("EAR", e_frame)
    #     #
    #     # cv.waitKey(1)
    #     # cv.destroyAllWindows()




