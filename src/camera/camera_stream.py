import cv2

class CameraStream:
    def __init__(self, camera_index=0):
        """
        Initialize the camera stream.
        """
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

    def get_frame(self):
        """
        Reads one frame from the camera.
        Returns:
            status (bool) - True if frame is valid
            frame (ndarray) - The image frame
        """
        status, frame = self.cap.read()
        return status, frame

    def show(self, window_name, frame):
        """
        Displays the frame in a window.
        """
        cv2.imshow(window_name, frame)

    def close(self):
        """
        Releases resources.
        """
        self.cap.release()
        cv2.destroyAllWindows()