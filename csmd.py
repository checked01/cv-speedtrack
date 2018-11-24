# imports
import cv2
import time
import threading
import math
import csv


# reads from the camera stream on a separate thread
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

    # start thread which will read from
    def start(self):
        self.stopped = False
        threading.Thread(target=self.update).start()

    def stop(self):
        self.stopped = True


# linear search to obtain the largest contour given a list
def get_greatest_contour(contours):
    largest_contour = None

    for c in contours:
        if largest_contour is None:
            largest_contour = c
            continue

        if cv2.contourArea(c) > cv2.contourArea(largest_contour):
            largest_contour = c

    return largest_contour

# video capture source
cap = cv2.VideoCapture("rtsp://10.66.6.101/11")
#cap = cv2.VideoCapture(0)

# start the threaded camera stream
t_stream = ThreadStream(cap)
t_stream.start()

# variables we will use in our program
pre_frame = None
previous_x_pos = 1
previous_y_pos = 1
pre_time = time.time()
start_time = time.time()

# open the logfile and define headers
with open("logs/" + str(round(start_time, 0)) + '_car_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['time', 'x', 'y', 'speed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# main processing loop
while True:
    # get the frame
    frame = t_stream.return_frame()

    if frame is not None:
        draw_frame = frame.copy()

        # apply a grayscale to the frame
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur
        blurred_frame = cv2.GaussianBlur(grayscaled_frame, (21, 21), 0)

        # if pre_frame has not yet been set, set it and loop back
        if pre_frame is None:
            pre_frame = grayscaled_frame
            continue

        # get the absolute difference between current frame and previous
        delta_frame = cv2.absdiff(pre_frame, blurred_frame)
        # apply a threshold to frame
        thresholded_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate frame to smooth, the number of iterations will increase smoothing
        dilated_frame = cv2.dilate(thresholded_frame, None, iterations=15)

        # find contours in frame
        found_contours = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_contours = found_contours[1]

        # get the greatest contour (largest shape in the frame), and track it
        selected_contour = get_greatest_contour(found_contours)

        if selected_contour is not None:
            # get a bounding box on the largest contour
            (x, y, w, h) = cv2.boundingRect(selected_contour)

            # draw a rectangle around the selected contour
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            averaged_x_pos = x + (w / 2)
            averaged_y_pos = y + (h / 2)

            delta_time = time.time() - pre_time
            delta_xt = (averaged_x_pos - previous_x_pos) / delta_time
            delta_yt = (averaged_y_pos - previous_y_pos) / delta_time

            velocity = math.sqrt(math.pow(previous_x_pos, 2) + math.pow(previous_y_pos, 2))

            previous_x_pos = averaged_x_pos
            previous_y_pos = averaged_y_pos

            # draw to the gui frame
            cv2.putText(draw_frame, "x: " + str(averaged_x_pos), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(draw_frame, "y: " + str(averaged_y_pos), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(draw_frame, "vx: " + str(delta_xt), (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(draw_frame, "vy: " + str(delta_yt), (10, 110), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            pre_time = time.time()

            # log to file
            with open("logs/" + str(round(start_time, 0)) + '_car_data.csv', 'a', newline='') as csvfile:
                fieldnames = ['time', 'x', 'y', 'speed']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                print(str(round(start_time - pre_time, 2)) + ", " + str(velocity))
                writer.writerow({'time': str(round(pre_time - start_time, 2)), 'x': averaged_x_pos, 'y': averaged_y_pos, 'speed': str(velocity)})

        # show the frame with gui applied
        cv2.imshow("draw_frame", draw_frame)

        pre_frame = blurred_frame

        # if the user presses "ESCAPE" on the window, exit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# release the thread stream and destroy windows, general cleanup
t_stream.stop()
cap.release()
cv2.destroyAllWindows()
