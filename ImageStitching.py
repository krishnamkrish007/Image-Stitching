import numpy as np
import cv2
import imutils.video


class Panorama:
    def __init__(self, cam_ids=[0],k=2,ratio=0.75,reprojThresh=4, n_matches=10, freq=100):
        #Get the camera IDs from user
        #camers ids has to be in left to right order
        self.cam_ids = cam_ids
        self.homography = [None]*(len(self.cam_ids)-1)
        self.k = k
        self.ratio = ratio
        self.reprojThresh=4
        self.count =0
        self.n_matches = n_matches
        self.freq = freq

    def camInit(self):
        #Initialize the video streams object for each camera
        self.VideoStreams = []
        for i in  self.cam_ids:
            self.VideoStreams.append(imutils.video.VideoStream(src=i).start())


    def read(self):
        #Read one frame from each camera in parallal and return them as an array
        frames = []
        for i in self.VideoStreams:
            frames.append(i.read())
        self.count+=1

        return frames
    def update(self,frames):
        
        frames_gray = []
        for frame in frames:
            frames_gray.append(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
        #using ORB for detection of features and descriptors
        descriptor = cv2.ORB_create()


        keyPoints =[]
        features =[]

        for frame in frames_gray:
            (kps,feat) = descriptor.detectAndCompute(frame,None)
            keyPoints.append(np.float32([kp.pt for kp in kps]))
            features.append(feat)


        matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = False)

        for i  in range(len(features)-1):

            rawMatches = matcher.knnMatch(features[i],features[i+1],self.k)
            matches = [m for m,n in rawMatches if m.distance<n.distance*self.ratio]
            if len(matches)>self.n_matches:
                ptsA = np.float32([keyPoints[i][m.queryIdx ] for m in matches])
                ptsB = np.float32([keyPoints[i+1][m.trainIdx] for m in matches])

                (H,status) = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,self.reprojThresh)

                self.homography[i]=H
            else:
                self.homography[i] = None

    def stitch(self):
        frames = self.read()
        self.update(frames)
        flag = False
        for H in self.homography:
            if H is not None:
                flag = True

        if not flag:
            return None
      

        width = sum([frame.shape[1] for frame in frames])
        height = sum([frame.shape[0] for frame in frames])
        result = cv2.warpPerspective(frames[0],self.homography[0],(width,height))
        for i in range(len(frames)-1):
            result[0:frames[i+1].shape[0],0:frames[i+1].shape[1]]= frames[i+1]
        return result


cams = PanoCams([0,2])
cams.camInit()

while True:
    stitched = cams.stitch()
    if stitched is not None:
        cv2.imshow("Result", stitched)
    if cv2.waitKey(1) & 0xFF==ord('q'): break
cv2.destroyAllWindows()



















