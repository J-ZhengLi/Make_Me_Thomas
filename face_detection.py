import cv2
from os import path
import numpy as np
from PIL import Image, ImageDraw

class FaceDetection():
    def __init__(self, path, classifier='Data/haarcascade_frontalface_default.xml'):
        self.imagePath = path
        self.classifier = classifier
        self.face = (0, 0, 0, 0)
        self.circle = (0, 0, 0)

    def detectFace(self, imageScale=1, scaleFactor=1.2, retryLimit=1):
        cascade = cv2.CascadeClassifier(self.classifier)
        o_image = cv2.imdecode(np.fromfile(self.imagePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if (imageScale != 1):
            self.image = cv2.resize(o_image, (int(o_image.shape[1]*imageScale), int(o_image.shape[0]*imageScale)))
        else:
            self.image = o_image

        grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.faces = cascade.detectMultiScale(grayImage, scaleFactor)

        if len(self.faces) < 1:
            # retry detection
            for c in range(retryLimit):
                if scaleFactor <= 1:
                    break
                else:
                    scaleFactor -= 0.5
                    faces = cascade.detectMultiScale(grayImage, scaleFactor)
            # if still no result
            print('Unable to find any face in this image')
        elif len(self.faces) > 1:
            # choose the largest one if multiple image has found
            max_val = 0
            max_idx = 0
            for f in range(len(self.faces)):
                if self.faces[f][2] > max_val:
                    max_val = self.faces[f][2]
                    max_idx = f
            self.face = self.faces[max_idx]
        else:
            self.face = self.faces[0]
        

    def getCroppedFace(self, radiusScale=1):
        maxSize = (0, 0)

        (x,y,w,h) = self.face
        if (w, h) > maxSize:
            maxSize = (w, h)
            self.circle = (int(x+w/2), int(y+h/2), int(w*radiusScale))

    def outputCroppedFaceImage(self, outFilename, radiusScale):
        if (self.face[2] != 0):
            if self.circle == (0, 0, 0):
                self.getCroppedFace(radiusScale)

            x = (self.circle[0] - self.circle[2])
            y = (self.circle[1] - self.circle[2])
            newImg = self.image[y:(y+2*self.circle[2]), x:(x+2*self.circle[2])]
            npImg = np.array(newImg)
            h, w, _ = newImg.shape
            alphaLayer = Image.new('L', (h, w), 0)
            draw = ImageDraw.Draw(alphaLayer)
            draw.pieslice([0,0,h,w],0,360,fill=255)
            npAlpha=np.array(alphaLayer)
            npImg=np.dstack((npImg,npAlpha))
            cv2.imwrite(outFilename, npImg)

    '''
    The next two functions are for Debug ONLY
    '''
    def circleFaceInImage(self, image, face=(), radiusScale=1):
        if face.any():
            cv2.circle(image, (int(face[0]+face[2]/2), int(face[1]+face[3]/2)), int(face[2]*radiusScale), (0,0,255))
            cv2.imshow('Face', image)
            cv2.waitKey()
        else:
            print('Cannot show the result image because no face was detected')
    
    def circleFacesInImage(self, image, faces=[], radiusScale=1):
        for face in faces:
            cv2.circle(image, (int(face[0]+face[2]/2), int(face[1]+face[3]/2)), int(face[2]*radiusScale), (255,0,0))
        cv2.imshow('Faces', image)
        cv2.waitKey()