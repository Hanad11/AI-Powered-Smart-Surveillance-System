import cv2
from camera.camera_stream import CameraStream
from face_detection.detector import FaceDetector

def main():
    # Initialize camera
    camera = CameraStream()

    # Initialize face detector
    detector = FaceDetector()

    while True:
        status, frame = camera.get_frame()

        if not status:
            print("Error: Could not read frame.")
            break

        # ----------------------------------------
        # FACE DETECTION
        # ----------------------------------------
        boxes = detector.detect_faces(frame)

        # Draw rectangles for each detected face
        for (top, right, bottom, left) in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show the output
        camera.show("Face Detection Test (Haar Cascade)", frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    camera.close()


if __name__ == "__main__":
    main()