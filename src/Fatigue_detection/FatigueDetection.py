
import cv2 as cv

from src.Fatigue_detection.submodules.YawnDetection import YawnDetection


class FatigueDetection:
    def __init__(self):
        self.yawn_detection = YawnDetection()


    def process_frame(self, frame):
        return self.yawn_detection.process_frame(frame)

    # def facial_expression_analysis(self):
    #     pass
    #
    # def eye_blink_detection(self):
    #     pass
    #
    # def head_pose_estimation(self):
    #     pass



if __name__ == '__main__':
    fd = FatigueDetection()

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frames found")
            exit()

        frame = cv.flip(frame,1)

        output_frame = fd.process_frame(frame)

        cv.imshow("Yawn detector", output_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

