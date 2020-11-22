import cv2
import imutils
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib

# Pre-trained car classifier
classifier_file = 'classifiers/cars.xml'

# Create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Read video
vc = cv2.VideoCapture("videos/normal-daytime/bridge_car_bus.mp4")
#vc = cv2.VideoCapture("videos/nighttime/berlin_night.mp4")


ct = CentroidTracker(maxDisappeared=10)

trackable_objects = {}
key = cv2.waitKey(1) & 0xFF
total_frames = 0

while True:                                                        # loop over frames from the video stream
    frame = vc.read()                                              # grab the current frame, then handle if we are using a
    frame = frame[1]                                               # VideoStream or VideoCapture object
    total_frames += 1

    if frame is None or key == ord("q"):                                              # check to see if we have reached the end of the stream
        break

    frame = imutils.resize(frame, width=500)                       # resize the frame (so we can process it faster)

    black_n_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # Convert to Grayscale

    rects = []
    #cars = car_tracker.detectMultiScale(black_n_white)             # detect cars

    #for (x, y, w, h) in cars:                                      # Draw rectangles around the cars:
    #    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    if total_frames % 3 == 0 or total_frames == 1:
        trackers = []
        cars = car_tracker.detectMultiScale(black_n_white)             # detect cars

        if len(cars) != 0 and frame is not None:
            for (x,y,w,h) in cars:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x,y,x+w,y+h)
                tracker.start_track(frame, rect)
                trackers.append(tracker)
    else:
        for tracker in trackers:
            tracker.update(frame)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items(): # loop over tracked objects
        to = trackable_objects.get(objectID, None)
        print(to)
        cv2.circle(frame, (centroid[0], centroid[1]), 7, (0, 255, 0), -1)

        if to is None: # if there is no existing trackable object, create one
            to = TrackableObject(objectID, centroid)
            trackable_objects[objectID] = to
            #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    print(trackable_objects)
    cv2.imshow("Frame", frame)      # show the output frame
    key = cv2.waitKey(1) & 0xFF

print("Vehicle Count = " + str(len(trackable_objects.keys())))