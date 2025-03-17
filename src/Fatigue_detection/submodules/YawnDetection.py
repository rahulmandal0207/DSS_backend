

import mediapipe.python.solutions as mp
import cv2 as cv
import numpy as np

class YawnDetection:
    def __init__(self):
        self.mp_face_mesh = mp.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.drawing_utils
        self.mp_drawing_style = mp.drawing_styles
        self.YAWN_THRESHOLD = 0.9


    def __mouth_aspect_ratio(self, mouth):
        top_lip = mouth[0]
        bottom_lip = mouth[1]
        left_corner = mouth[2]
        right_corner = mouth[3]

        A = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
        B = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))

        mar = A / B
        return mar


    def detect_yawn(self, frame):
        rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
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
                print(mar)
                if mar < self.YAWN_THRESHOLD:
                    return True, face_landmark
        return False, None

    def  draw_landmarks(self, frame, landmarks):
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_style.get_default_face_mesh_contours_style()
        )
        # for idx in [61, 291, 0, 17]:
        #     landmark = landmarks.landmark[idx]
        #     x = int(landmark.x * frame.shape[1])
        #     y = int(landmark.y * frame.shape[0])
        #     cv.putText(frame, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, .3,(0, 0, 255), 1)  # Draw a red circle on the landmark

    def process_frame(self, frame):
        yawn, landmarks = self.detect_yawn(frame)

        if yawn:
            cv.putText(frame, "Yawing Detected!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.draw_landmarks(frame, landmarks)
        else:
            cv.putText(frame, "No Yawn!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame


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




