import cv2 as cv

# Car Video

video = cv.VideoCapture('Tesla Dashcam.mp4')
video2 = cv.VideoCapture('Cars_Pedestrians_Trim.mp4')

# Pre-trained car and pedestrian classifiers
car_tracker_file = ('cars3.xml')
pedestrian_tracker_file = ('haarcascade_fullbody.xml')

# create car classifier
car_tracker = cv.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv.CascadeClassifier(pedestrian_tracker_file)




# Always Run
while True:

    #read current frame
    (read_successful, frame) = video2.read()

    # Safe coding
    if read_successful:
        #we must convert car images to grayscale(it is needed for haar cascade)
        grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        break

    
    
    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in cars:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2 )
    
    for (x, y, w, h) in pedestrians:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2 )

    
    
    # Display the image with the cars spotted
    cv.imshow('Clever Car and Pedestrian Detector', frame)

    # Dont autoclose (Wait here in the code until key is pressed) 
    key = cv.waitKey(1)

    # if q key is pressed
    if key == 81 or key == 113:
        
        break 

# Release the vidoCapture object because it is always reading.
video2.release()   

    
    

    





  

