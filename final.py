import cv2
import imutils
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib
import sys

def createTracker(tracked_objs, frame):
    trackers_list = []
    if len(tracked_objs) != 0 and frame is not None:
        for (x,y,w,h) in tracked_objs:
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x,y,x+w,y+h)
            tracker.start_track(frame, rect)
            trackers_list.append(tracker)
    return trackers_list

def updateTrackers(trackers):
    rects = []
    for tracker in trackers:
        tracker.update(frame)
        pos = tracker.get_position()

        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())

        rects.append((startX, startY, endX, endY))
    return rects

def defineTrackableObjects(objects, dict, color):
    for (objectID, centroid) in objects.items(): # loop over tracked objects
        to = dict.get(objectID, None)
        if color == 0:
            cv2.circle(frame, (centroid[0], centroid[1]), 7, (0, 255, 0), -1)
        if color == 1:
            cv2.circle(frame, (centroid[0], centroid[1]), 7, (255, 0, 0), -1)
        if color == 2:
            cv2.circle(frame, (centroid[0], centroid[1]), 7, (0, 0, 255), -1)
        if color == 3:
            cv2.circle(frame, (centroid[0], centroid[1]), 7, (255, 0, 255), -1)

        if to is None: # if there is no existing trackable object, create one
            to = TrackableObject(objectID, centroid)
            dict[objectID] = to
    return dict


# Read video input
args = sys.argv
filename = args[1]
vc = cv2.VideoCapture(filename)

# Pre-trained cascade classifiers
car_classifier_file = 'new_classifiers/cars.xml'
ped_classifier_file = 'new_classifiers/pedestrian.xml'
bike_classifier_file = 'new_classifiers/two_wheeler.xml'
bus_classifier_file = 'new_classifiers/bus_front.xml'

# Create classifiers
car_tracker = cv2.CascadeClassifier(car_classifier_file)
ped_tracker = cv2.CascadeClassifier(ped_classifier_file)
bike_tracker = cv2.CascadeClassifier(bike_classifier_file)
bus_tracker = cv2.CascadeClassifier(bus_classifier_file)

# Create centroid tracker
ct_cars = CentroidTracker(maxDisappeared=10)
ct_peds = CentroidTracker(maxDisappeared=10)
ct_bike = CentroidTracker(maxDisappeared=10)
ct_bus = CentroidTracker(maxDisappeared=10)

# Create dictionary of unique objects
trackable_cars = {}
trackable_peds = {}
trackable_bikes = {}
trackable_bus = {}

# Housekeeping
key = cv2.waitKey(1) & 0xFF      # get keystrokes
total_frames = 0                 # start total frame count

while True:                                                        # loop over frames from the video stream
    frame = vc.read()                                              # grab the current frame, then handle if we are using a
    frame = frame[1]
    total_frames += 1

    if frame is None or key == ord("q"):                           # check to see if we have reached the end of the stream
        break

    frame = imutils.resize(frame, width=500)                       # resize the frame (so we can process it faster)

    black_n_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # Convert to Grayscale

    car_rects = []
    ped_rects = []
    bike_rects = []
    bus_rects = []

    # Only perform new object detection every so often number of frames
    if total_frames % 6 == 0 or total_frames == 1:
        cars = car_tracker.detectMultiScale(black_n_white)          # detect objects
        peds = ped_tracker.detectMultiScale(black_n_white)
        bikes = bike_tracker.detectMultiScale(black_n_white)
        busses = bus_tracker.detectMultiScale(black_n_white)

        car_trackers = createTracker(cars, frame)                   # create trackers for objects
        ped_trackers = createTracker(peds, frame)
        bike_trackers = createTracker(bikes, frame)
        bus_trackers = createTracker(busses, frame)
    else:
        car_rects = updateTrackers(car_trackers)
        ped_rects = updateTrackers(ped_trackers)
        bike_rects = updateTrackers(bike_trackers)
        bus_rects = updateTrackers(bus_trackers)


    # use the centroid tracker to associate old object
    # centroids with newly computed object centroids
    car_objects = ct_cars.update(car_rects)
    ped_objects = ct_peds.update(ped_rects)
    bike_objects = ct_bike.update(bike_rects)
    bus_objects = ct_bus.update(bus_rects)

    # Update dictionaries with unique objects
    trackable_cars = defineTrackableObjects(car_objects, trackable_cars, 0)
    trackable_peds = defineTrackableObjects(ped_objects, trackable_peds, 1)
    trackable_bikes = defineTrackableObjects(bike_objects, trackable_bikes, 2)
    trackable_bus = defineTrackableObjects(bus_objects, trackable_bus, 3)


    cv2.imshow("Frame", frame)      # show the output frame
    key = cv2.waitKey(1) & 0xFF

print("Vehicle Count = " + str(len(trackable_cars.keys())))
print("Pedestrian Count = " + str(len(trackable_peds.keys())))
print("Bike Count = " + str(len(trackable_bikes.keys())))
print("Bus Count = " + str(len(trackable_bus)))