import cv2
import time
import threading


class ThreadStream:
    def __init__(self, stream_src):
        self.stream = stream_src
        self.frame = None
        self.stopped = True

    def return_frame(self):
        return self.frame

    def update(self):
        while True:
            if self.stopped:
                return

            (ret, self.frame) = self.stream.read()
            time.sleep(.05)

    def start(self):
        self.stopped = False
        threading.Thread(target=self.update).start()

    def stop(self):
        self.stopped = True


# cascade source and classifier
cascade_src = 'classifiers/cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

# read from IP camera
cap = cv2.VideoCapture("rtsp://10.66.6.101/11")
#cap = cv2.VideoCapture("video1.avi")

t_stream = ThreadStream(cap)
t_stream.start()

while True:
    frame = t_stream.return_frame()

    if frame is not None:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roi_split = grayscale_frame[0:900, 0:300]

        cars = car_cascade.detectMultiScale(roi_split, 1.1, 1)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if w > 50 and h > 50:
                print("Car Detected!")

        cv2.imshow('video', frame)

        if cv2.waitKey(33) == 27:
            t_stream.stop()
            break

