import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self,
                 mode = False,
                 complexity = 1,
                 smooth = True,
                 enableSeg = False,
                 smoothSeg = True,
                 detectCon = 0.5,
                 trackCon = 0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,
                                     self.complexity,
                                     self.smooth,
                                     self.enableSeg,
                                     self.smoothSeg,
                                     self.detectCon,
                                     self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (80, 255, 222), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('PoseVideos/dance.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            print(lmList[14])
            #cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 255, 0), cv2.FILLED)

        # calculate FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # visualize output
        cv2.putText(img, f"FPS:{str(int(fps))}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (80, 255, 222), 2)
        cv2.imshow('PoseEstimation', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
