import cv2

# Image
img_file = 'test_image.jpg'

# Pre-trained car classifier
car_classifier_file = 'classifiers/cars.xml'
ped_classifier_file = 'classifiers/pedestrians.xml'
#bike_classifier_file = 'classifiers/bike.xml' #backup
bus_classifier_file = 'classifiers/bus.xml'
bicycle_classifier_file = 'classifiers/bicycle.xml'
motorbike_classifier_file = 'classifiers/motorbike.xml'


# Create opencv image
img = cv2.imread(img_file)

# Create classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
ped_tracker = cv2.CascadeClassifier(ped_classifier_file)
#bike_tracker = cv2.CascadeClassifier(bike_classifier_file)
bus_tracker = cv2.CascadeClassifier(bus_classifier_file)
bicycle_tracker = cv2.CascadeClassifier(bicycle_classifier_file)
motorbike_tracker = cv2.CascadeClassifier(motorbike_classifier_file)

# Convert to Grayscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#print(black_n_white)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)
pedestrians = ped_tracker.detectMultiScale(black_n_white)
buses = bus_tracker.detectMultiScale(black_n_white)
bicycles = bicycle_tracker.detectMultiScale(black_n_white)
motorbikes = motorbike_tracker.detectMultiScale(black_n_white)

print(pedestrians)

# Draw rectangles around the cars:
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

for (x, y, w, h) in pedestrians:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

for (x, y, w, h) in buses:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

for (x, y, w, h) in bicycles:
    cv2.rectangle(img, (x,y), (x+w, y+h), (51, 199, 95), 2)

for (x, y, w, h) in motorbikes:
    cv2.rectangle(img, (x,y), (x+w, y+h), (102, 0, 204), 2) 

cv2.imshow('Car and Pedestrian Detector', img)
cv2.waitKey()
