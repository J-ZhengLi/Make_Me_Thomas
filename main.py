from face_detection import FaceDetection
import numpy as np
from PIL import Image
from sys import argv

def generateFaceImage(inputFp, outputFp, scale):
    fd = FaceDetection(inputFp)
    fd.detectFace(scaleFactor=1.2, retryLimit=5)
    fd.outputCroppedFaceImage(outputFp, scale)

def main():
    cropScale = 0.55
    inputFp = 'test/dt.jpg'

    if len(argv) == 2:
        inputFp = argv[1]
    elif len(argv) > 2:
        inputFp = argv[1]
        cropScale = float(argv[2])

    faceRadius = 238
    filePath = 'testCrop.png'

    generateFaceImage(inputFp, filePath, cropScale)
    
    thomasImg = np.array(Image.open('Data/thomas.jpg'))
    swapImg = np.array(Image.open(filePath).resize((faceRadius, faceRadius)))

    resultImg = np.copy(thomasImg)
    # replace thomas face to the input image
    for i in range(faceRadius):
        for j in range(faceRadius):
            if swapImg[i, j][3] != 0:
                # Manually convert to grayscale
                resultImg[167+i, 358+j] = np.dot(swapImg[i, j][:3], [0.3, 0.6, 0.15])
    
    result = Image.fromarray(resultImg, 'RGB')
    result.save('output.jpg')
    result.show()
    

if __name__ == "__main__":
    main()