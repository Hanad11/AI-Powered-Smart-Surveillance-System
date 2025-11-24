import cv2

def resize_frame(frame, scale=0.5):
    """
    Resize a frame by a scaling factor.
    """
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv2.resize(frame, (width, height))

def convert_to_rgb(frame):
    """
    Convert BGR (OpenCV default) to RGB.
    Needed for face recognition libraries.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def convert_to_gray(frame):
    """
    Convert BGR to Grayscale.
    Useful for some detection methods.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def flip_horizontal(frame):
    """
    Flip frame horizontally (mirror effect).
    """
    return cv2.flip(frame, 1)

def crop_face(frame, box):
    """
    Crop a face region from the frame using the bounding box.
    box format: (top, right, bottom, left)
    """
    top, right, bottom, left = box
    return frame[top:bottom, left:right]

def save_frame(frame, filename="snapshot.jpg"):
    """
    Save a frame to disk.
    """
    cv2.imwrite(filename, frame)