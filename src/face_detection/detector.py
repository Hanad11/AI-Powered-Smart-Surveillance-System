import cv2
import os

class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5):
        """
        Initialize the Haar Cascade face detector.
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            raise Exception("Haar Cascade file not found.")

        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, frame):
        """
        Detect faces and return bounding boxes in the format:
        (top, right, bottom, left)
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors
        )

        boxes = []
        for (x, y, w, h) in detections:
            top = y
            right = x + w
            bottom = y + h
            left = x
            boxes.append((top, right, bottom, left))

        return boxes
