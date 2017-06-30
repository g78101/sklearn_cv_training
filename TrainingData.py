from sklearn import svm
import numpy as np
import cv2
import os

myTrainingDataDir = 'TrainingData'

def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Avoid to add hidden file (MacOSX)
            if filename.startswith('.') == False:
              # Join the two strings in order to form the full filepath.
              filepath = os.path.join(root, filename)
              file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

lists = get_filepaths("%s/%s"%(os.path.dirname(os.path.abspath(__file__)),myTrainingDataDir))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001,C=100, kernel='poly')

trainArrry = None
targetArray = np.ndarray([0],int)

print 'Prepare TrainingData and Target'
for url in lists:
  image = cv2.imread(url,0)
  # Get image.size to init trainArray
  if trainArrry is None:
    trainArrry = np.ndarray([0,image.size],int)
  # Add TrainingData
  trainArrry = np.vstack((trainArrry,image.flatten()))
  # Add Target
  targetArray = np.append(targetArray,os.path.basename(url).split('_')[0])
# Check trainArrry and targetArray
print trainArrry.shape
print targetArray.shape
print 'Training'
clf.fit(trainArrry,targetArray)
print 'Finish'
from sklearn.externals import joblib
# Save result
joblib.dump(clf,"result.pkl")