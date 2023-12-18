import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

Reference_img = cv2.imread("face.jpg")

if Reference_img is None:
    print("Error loading the image. Please check the file path or format.")
else:
    cv2.imshow('Window Name', Reference_img)  # Display the image with a window name
    cv2.waitKey(0)  # Wait for any key press to close the window
    cv2.destroyAllWindows()

def check_face(frame):
    global face_match
    try:
        if Reference_img is not None:
            result = DeepFace.verify(frame, Reference_img)
            if result['verified']:
                face_match = True
            else:
                face_match = False
    except Exception as e:
        print("Error:", e)
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception as e:
                print("Error:", e)
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NOMatch!", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
