import cv2

path = './video_right.mp4'

def get_camera_frame():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible or not found.")
    else:
        print("Camera is working! Press 'q' to quit.")
        while True:
            ret, img = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            cv2.imshow("Camera Test", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()