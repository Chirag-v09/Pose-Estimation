import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('3.mp4')

detect = pm.PoseDetector()
ptime = 0

save_name = "PoseDetectionWithRightElbow.avi"
fps = 25
width = int(cap.get(3))
height = int(cap.get(4))
output_size = (width, height)
out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'XVID'), fps, output_size)

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
    # To calculate frames per second
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    out.write(img)

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    cv2.imshow('video', img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
