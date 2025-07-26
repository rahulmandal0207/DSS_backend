import pandas as pd
import cv2 as cv

from src.Fatigue_detection.FatigueDetection import FatigueDetection
from src.Model.Model import Model
from playsound import playsound

fd = FatigueDetection()
model =  Model()

data = "../resources/data/output/top_20.csv"
df = pd.read_csv(data)

model.preprocessing(df)
model.train()
model.predict()

cap = cv.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("No frame")
        exit()

    y_frame, e_frame, mar, lear, rear = fd.process_frame(frame)


    result = model.predict_one(mar, lear, rear)

    if result[0] == 1:
        playsound('../resources/beep-warning-6387.mp3')

    print(result[0])

    if y_frame is not None:
        cv.imshow("Yawn frame", y_frame)

    if e_frame is not None:
        cv.imshow("Eye frame", e_frame)


    # cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv.destroyAllWindows()






