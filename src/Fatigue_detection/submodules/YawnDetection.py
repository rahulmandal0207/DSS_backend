
from mediapipe.python import solutions
import cv2 as cv
import numpy as np

class YawnDetection:
    def __init__(self, yawn_threshold=0.9):

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

    def detect_yawn(self, face_landmarks):

        mouth_points = [
            face_landmarks.landmark[61],
            face_landmarks.landmark[291],
            face_landmarks.landmark[0],
            face_landmarks.landmark[17],
        ]
        mar = self.__mouth_aspect_ratio(mouth_points)
        if mar < self.YAWN_THRESHOLD:
            return True, face_landmarks, mar
        return False, face_landmarks, None

    def  draw_landmarks(self, frame,is_yawn, landmarks):
        lips_points = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        for idx in lips_points:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            # cv.putText(frame, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, .3,(0, 0, 255), 1)  # Draw a red circle on the landmark
            if is_yawn:
                cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            else:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def process_frame(self, frame, face_landmarks):
        is_yawn, landmarks, mar = self.detect_yawn(face_landmarks)
        if is_yawn:
            cv.putText(frame, "Yawing Detected!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.draw_landmarks(frame, is_yawn, landmarks)
        else:
            cv.putText(frame, "No Yawn!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.draw_landmarks(frame, is_yawn,landmarks)


        return frame, mar


if __name__ == '__main__':

    mp_face_mesh = solutions.face_mesh
    mp_face = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    )

    yd = YawnDetection()
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No frame")
            exit()

        frame = cv.flip(frame,1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = mp_face.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                output_frame = yd.process_frame(frame, face_landmarks)

        cv.imshow("Frame",output_frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
    else:
        print("No source found.")

    cap.release()
    cv.destroyAllWindows()




