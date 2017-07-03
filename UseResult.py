from sklearn.externals import joblib
import numpy as np
import cv2

imageName = "1.jpg"

# Load result.pkl
clf =joblib.load("result.pkl")

im = cv2.imread(imageName)

# Preprocessing Image
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)

# Get rectangles contains each contour
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for (x,y,w,h) in rects:
  # Ignore small region
  if w*h > 1500:
    # Draw the rectangles
    cv2.rectangle(im, (x-(h-w)/2, y), (x+h, y+h), (255, 255, 0), 3)  
    # Get roi from image
    roi = im_th[y:y+h, x:x+w]
    length = max(w,h)
    # Put in square image
    squareImage = np.zeros([length,length], dtype=np.uint8)
    squareImage[(length-h)/2:(length-h)/2+h,(length-w)/2:(length-w)/2+w]=roi
    # Resize the square mat
    squareImage = cv2.resize(squareImage, (20, 20))
    squareImage = cv2.dilate(squareImage, None)
    #Prediction the square mat
    result = clf.predict(squareImage.flatten().reshape(1,-1))
    cv2.putText(im, str(result[0]), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Result", im)
cv2.waitKey()