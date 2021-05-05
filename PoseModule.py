"""
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
"""

import cv2
import time
import mediapipe as mp


class PoseDetector():

    def __init__(self, static_image_mode=False, upper_body_only=False, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.upper_body_only, self.smooth_landmarks,
                                     self.min_detection_confidence, self.min_tracking_confidence)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)
        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    cap = cv2.VideoCapture('1.mp4')

    detect = PoseDetector()
    ptime = 0
    while cap.isOpened():

        success, img = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q') or success == False:
            cap.release()
            cv2.destroyAllWindows()
            break

        img = detect.find_pose(img)
        lm_list = detect.find_position(img, False)
        if len(lm_list) != 0:
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        # print(lm_list)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
        cv2.imshow('video', img)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
