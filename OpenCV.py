import numpy as np
import cv2
import matplotlib.pyplot as plt

#  Loading the image to be tested
test_image = cv2.imread('image3.jpg')

# Converting to greyscale as opencv expects detector takes in input gray scale images
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
font = cv2.FONT_HERSHEY_SIMPLEX
# Displaying greyscale image
plt.imshow(test_image_gray, cmap='gray')
plt.show()


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Face detection information
haar_cascade_face = cv2.CascadeClassifier('thing.xml')
haar_cascade_eye = cv2.CascadeClassifier('thing2.xml')
# scalefactor: In a group photo, there may be some faces which are near the camera than others.
# Naturally, such faces would appear more prominent than the ones behind. This factor compensates for that.
# minNeighbors: This parameter specifies the number of neighbors a rectangle should have to be called a face.

def detect_faces(cascade1, cascade2, test_image, scaleFactor=1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image

    # convert the test image to gray scale as opencv face detector expects gray images
    grey_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    n = 9
    # Applying the haar classifier to detect faces
    faces_rect = cascade1.detectMultiScale(grey_image, scaleFactor=scaleFactor, minNeighbors=8)
    eyes_rect = cascade2.detectMultiScale(grey_image, scaleFactor=scaleFactor, minNeighbors=n)
    ar = []
    while len(eyes_rect) < (len(faces_rect) * 2):
        eyes_rect = cascade2.detectMultiScale(test_image, scaleFactor=scaleFactor, minNeighbors=n)
        n -= 1
    # Gets the width and and height of the image so it can draw the rectangle
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(image_copy, 'Face', ((x+w-60), (y+h+30)), font, 0.7, (255, 0, 240), 2)
        temp = [x, y, w, h]
        ar.append(temp)
    for (x, y, w, h) in eyes_rect:
        for i in range(0, len(ar)):
            if (x > (ar[i][0]) and x < (ar[i][0] + ar[i][2]) and y > ar[i][1] and y < (ar[i][1] + ar[i][2])):
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 5)
                cv2.putText(image_copy, 'Eye', (x, (y+h+20)), font, 0.6, (0, 0, 255), 2)
    return image_copy


# call the function to detect faces
faces = detect_faces(haar_cascade_face, haar_cascade_eye, test_image)

# convert to RGB and display image
plt.imshow(convertToRGB(faces))
plt.show()

###################################################################################Video#########################################################################################################################################


cap = cv2.VideoCapture(0)

def detect_facesVid(cascade1, cascade2, vid, scaleFactor=1.1):
    # Applying the haar classifier to detect faces
    n = 18
    faces_rect = cascade1.detectMultiScale(vid, scaleFactor=scaleFactor, minNeighbors=8)
    eyes_rect = cascade2.detectMultiScale(vid, scaleFactor=scaleFactor, minNeighbors=n)
    while len(eyes_rect) > (len(faces_rect)*2):
        eyes_rect = cascade2.detectMultiScale(vid, scaleFactor=scaleFactor, minNeighbors=n)
        n += 3
    ar = []
    vid = cv2.cvtColor(vid, cv2.COLOR_GRAY2BGR)
    # Gets the width and and height of the image so it can draw the rectangle
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(vid, 'Face', ((x+w-60), (y+h+30)), font, 0.7, (255, 0, 240), 2)
        temp = [x, y, w, h]
        ar.append(temp)
    for (x, y, w, h) in eyes_rect:
        for i in range(0, len(ar)):
            if (x > (ar[i][0]) and x < (ar[i][0] + ar[i][2]) and y > ar[i][1] and y < (ar[i][1] + ar[i][2])):
                cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 5)
                cv2.putText(vid, 'Eye', (x, (y+h+20)), font, 0.6, (0, 0, 255), 2)
    return vid

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', detect_facesVid(haar_cascade_face, haar_cascade_eye, gray))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()