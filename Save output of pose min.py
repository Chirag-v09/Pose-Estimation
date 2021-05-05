
import cv2
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('3.mp4')

ptime = 0

save_name = "PoseDetection.avi"
fps = 25
width = int(cap.get(3))
height = int(cap.get(4))
output_size = (width, height)
out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'XVID'), fps, output_size)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    # To calculate frames per second
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    out.write(img)
    # out.write(cv2.resize(img, output_size))

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    cv2.imshow('video', img)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()

