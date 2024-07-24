import cv2
from tkinter import filedialog
from emotion_detection import crop_face, detect_emotions
from ui import display_image

def choose_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    emotion = "No face detected"
                else:
                    emotion = "Unknown"
                    for (x, y, w, h) in faces:
                        cropped_face = crop_face(image, x, y, w, h, target_size=(48, 48))
                        detected_emotion = detect_emotions(cropped_face)
                        emotion = detected_emotion
                
                display_image(image, emotion)
            else:
                print("Error loading image")
    except Exception as e:
        print(f"Error in choose_image: {e}")
