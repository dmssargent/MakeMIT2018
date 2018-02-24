#!/usr/bin/env python
import freenect
import cv2
import frame_convert2

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray_alpha = None
gray_beta = None
gray_delta = None


def detectFace(img, cascade):
    for scale in [float(i)/10 for i in range(11, 15)]:
        for neighbors in range(1,5):
            rects = cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=neighbors,
                                             minSize=(20, 20), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

            print 'scale: %s, neighbors: %s, len rects: %d' % (scale, neighbors, len(rects))
	    if len(rects) > 0:
                return rects
    return []

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)	

def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

def get_processed_frame():
    return get_motion_processed_frame()

def get_face_processed_frame():
    frame = get_video()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray)
    faces = detect(gray, face_cascade)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    return gray

def get_motion_processed_frame():
    gray_delta = gray_alpha
    gray_alpha = gray_beta
    gray_beta = cv2.cvtColor(get_video(), cv2.COLOR_BGR2GRAY)

    return diffImg(gray_delta, gray_alpha, gray_beta)

#Read 3 frames
gray_delta = cv2.cvtColor(get_video(), cv2.COLOR_BGR2GRAY)
gray_alpha = cv2.cvtColor(get_video(), cv2.COLOR_BGR2GRAY)
gray_beta = cv2.cvtColor(get_video(), cv2.COLOR_BGR2GRAY)

while 1:
    cv2.imshow('Depth', get_depth())
    cv2.imshow('Video', get_processed_frame())
    if cv2.waitKey(10) == 27:
        break
