import cv2
import time
import PoseModule as pm


def main():
    cap = cv2.VideoCapture('PoseVideos/soccer_1.mp4')
    pTime = 0
    detector = pm.poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 8, (186, 220, 108), 3)
        cv2.imshow('PoseEstimation', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()