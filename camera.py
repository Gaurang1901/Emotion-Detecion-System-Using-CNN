import cv2
from emotion_detection import crop_face, detect_emotions

def start_camera():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Error opening video capture")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Error reading frame from camera")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cropped_face = crop_face(frame, x, y, w, h, target_size=(48, 48))
                label = detect_emotions(cropped_face)
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            cv2.imshow("EMOTION DETECTION", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in start_camera: {e}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
