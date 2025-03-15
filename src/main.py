

import mediapipe.python.solutions as mp
import cv2 as cv
import numpy as np


mp_face_mesh = mp.face_mesh
mp_drawing = mp.drawing_utils
mp_drawing_style = mp.drawing_styles
YAWN_THRESHOLD = 0.2

cap = cv.VideoCapture(0)

def mouth_aspect_ratio(mouth):
    top_lip = mouth[0]
    bottom_lip = mouth[1]
    left_corner = mouth[2]
    right_corner = mouth[3]

    A = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
    B = np.linalg.norm(np.array([left_corner.x, left_corner.y]) - np.array([right_corner.x, right_corner.y]))

    mar = A / B
    return mar


def yawning_detection(frame):
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mouth_points = [
                face_landmark.landmark[61],
                face_landmark.landmark[291],
                face_landmark.landmark[0],
                face_landmark.landmark[17],
            ]
            mar = mouth_aspect_ratio(mouth_points)
            if mar > YAWN_THRESHOLD:
                return True, face_landmark
    return False, None

def  draw_landmarks(frame, landmarks):
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_style.get_default_face_mesh_contours_style()
    )

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("No frame")
        exit()

    frame = cv.flip(frame,1)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ).process(frame)

    yawn, landmarks = yawning_detection(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    if yawn:
        cv.putText(frame, "Yawing Detected!", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        draw_landmarks(frame, landmarks)


    cv.imshow("Frame",frame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break
else:
    print("No source found.")

cap.release()
cv.destroyAllWindows()




