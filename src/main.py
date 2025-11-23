from camera.camera_stream import CameraStream
import cv2

def main():
    camera = CameraStream()

    while True:
        status, frame = camera.get_frame()

        if not status:
            print("Error: Could not read frame.")
            break

        camera.show("Camera Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.close()

if __name__ == "__main__":
    main()
