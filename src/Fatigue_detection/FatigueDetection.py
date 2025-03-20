
import cv2 as cv

from src.Fatigue_detection.submodules.EyeClosureDetection import EyeClosureDetection
from src.Fatigue_detection.submodules.YawnDetection import YawnDetection


class FatigueDetection:
    def __init__(self):
        self.yawn_detection = YawnDetection()
        self.eye_closure_detection = EyeClosureDetection()


    def process_frame(self, frame):
        yawn = self.yawn_detection.process_frame(frame.copy())
        eye = self.eye_closure_detection.process_frame(frame.copy())
        return yawn, eye


if __name__ == '__main__':
    fd = FatigueDetection()

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frames found")
            exit()

        frame = cv.flip(frame,1)

        yawn_frame, eye_closure_frame = fd.process_frame(frame)

        cv.imshow("Yawn detector", yawn_frame)
        cv.imshow("Eye closure detector", eye_closure_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

