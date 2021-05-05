'''
STATIC_IMAGE_MODE
If set to false, the solution treats the input images as a video stream.
It will try to detect the most prominent person in the very first images, and upon a successful detection further
localizes the pose landmarks.
In subsequent images, it then simply tracks those landmarks without invoking another detection until it loses track,
on reducing computation and latency. If set to true, person detection runs every input image,
ideal for processing a batch of static, possibly unrelated, images. Default to false.

UPPER_BODY_ONLY
If set to true, the solution outputs only the 25 upper-body pose landmarks.
Otherwise, it outputs the full set of 33 pose landmarks.
Note that upper-body-only prediction may be more accurate for use cases where the lower-body parts are mostly out of view.
Default to false.

SMOOTH_LANDMARKS
If set to true, the solution filters pose landmarks across different input images to reduce jitter,
but ignored if static_image_mode is also set to true. Default to true.

MIN_DETECTION_CONFIDENCE
Minimum confidence value ([0.0, 1.0]) from the person-detection model for the detection to be considered successful.
Default to 0.5.

MIN_TRACKING_CONFIDENCE
Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the pose landmarks to be considered tracked
successfully, or otherwise person detection will be invoked automatically on the next input image.
Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency.
Ignored if static_image_mode is true, where person detection simply runs on every image. Default to 0.5.
'''
import cv2
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('2.mp4')

ptime = 0
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

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    cv2.imshow('video', img)
    cv2.waitKey(1)
