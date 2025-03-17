import dlib
import cv2

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the pre-trained shape predictor model for facial landmarks
predictor = dlib.shape_predictor("../resources/shape_predictor_68_face_landmarks.dat")

# Load an image using OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    # Convert the image to grayscale (dlib requires grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Loop over each face detected
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Loop over the 68 facial landmarks and draw them on the image
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle at each landmark
            cv2.putText(image,str(n),(x,y), cv2.FONT_HERSHEY_SIMPLEX, .4,(0,255,0),1)
    # Display the image with facial landmarks
    cv2.imshow("Facial Landmarks", image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()