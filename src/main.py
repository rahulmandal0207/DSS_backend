
import cv2 as cv
import mediapipe.python.solutions as mp

mp_face_mesh = mp.face_mesh
mp_face = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_tracking_confidence=.5,
    min_detection_confidence=.5
)
mp_drawing = mp.drawing_utils
mp_drawing_styles = mp.drawing_styles



cap = cv.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("No frame found")

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    top_lip = [61,  39,  0, 269, 291]
    bottom_lib = [181,  17, 405, 291]
    results = mp_face.process(rgb_frame)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            # )
            for idx in [ 61,  39,  0, 269, 291,
                181,  17, 405, 291
                         ]:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv.putText(frame, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0),
                           1)  # Draw a red circle on the landmark

    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break


