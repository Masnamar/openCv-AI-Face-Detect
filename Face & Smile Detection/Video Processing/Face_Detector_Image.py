import cv2 as cv
from random import randrange

# Lets load some pre-trained data on face frontals from opencv (haar cascade algorithm
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 

# Choose an image to detect faces
img = cv.imread('Face.jpg')
img1 = cv.imread('Muzo.PNG') # imread function reads images.
img2 = cv.imread('Various_faces.jpg')

# We must convert images to grayscale
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Lets detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img) #detectMukltiscale detects objects of different of size in the input image.

# print(face_coordinates) #we can see particular face coordinates in bash

# Lets draw rectangles around the frontal face
for (x, y, w, h) in face_coordinates:    
    cv.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)






cv.imshow('Clever Face Detector', img) # we show the image on the screen
cv.waitKey(0)

                   




    
