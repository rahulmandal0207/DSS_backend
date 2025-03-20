
import mediapipe.python.solutions as mp
import cv2 as cv
import numpy as np

class YawnDetection:
    def __init__(self, yawn_threshold=0.9):
        self.mp_face_mesh = mp.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.drawing_utils
        self.mp_drawing_style = mp.drawing_styles
        self.YAWN_THRESHOLD = yawn_threshold


    def __mouth_aspect_ratio(self, mouth):
        top_lip = mouth[0]
        bottom_lip = mouth[1]
        left_corner = mouth[2]
        right_corner = mouth[3]

        A = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
        B = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))

        mar = A / B
        return mar

    def detect_yawn(self, frame_yawn):

        rgb_frame = cv.cvtColor(frame_yawn, cv.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            for face_landmark in results.multi_face_landmarks:

                mouth_points = [
                    face_landmark.landmark[61],
                    face_landmark.landmark[291],
                    face_landmark.landmark[0],
                    face_landmark.landmark[17],
                ]

                mar = self.__mouth_aspect_ratio(mouth_points)
                if mar < self.YAWN_THRESHOLD:
                    return True, face_landmark
        return False, None

    def  draw_landmarks(self, frame_yawn, landmarks):
        for idx in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,146, 91, 181, 84, 17, 314, 405, 321, 375, 291]:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_yawn.shape[1])
            y = int(landmark.y * frame_yawn.shape[0])
            # cv.putText(frame, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, .3,(0, 0, 255), 1)  # Draw a red circle on the landmark
            cv.circle(frame_yawn, (x, y), 2, (0, 0, 255), -1)


    def process_frame(self, frame_yawn):
        yawn, landmarks = self.detect_yawn(frame_yawn)

        if yawn:
            cv.putText(frame_yawn, "Yawing Detected!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.draw_landmarks(frame_yawn, landmarks)
        else:
            cv.putText(frame_yawn, "No Yawn!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame_yawn


if __name__ == '__main__':

    yd = YawnDetection()
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frame")
            exit()

        frame = cv.flip(frame,1)

        output_frame = yd.process_frame(frame)

        cv.imshow("Frame",output_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
    else:
        print("No source found.")

    cap.release()
    cv.destroyAllWindows()




