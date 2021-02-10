import cv2 as cv

# Car Image
img_file = 'Car3.JPG'
video = cv.VideoCapture('Tesla Dashcam.mp4')

# Pre-trained car classifier
classifier_file = 'cars3.xml'

# Lets create opencv image
img = cv.imread(img_file) # reads all of the pixel data in the image file and store them multi-dimensional array

#we must convert car images to grayscale(it is needed for haar cascade)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(grayscale_img)



for (x, y, w, h) in cars:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255),2)
    





# Display the image with the cars spotted
cv.imshow('Clever Car and Pedestrian Detector',img)

# Dont autoclose (Wait here in the code until key is pressed)
key = cv.waitKey(0) 

















print("Code Finish")