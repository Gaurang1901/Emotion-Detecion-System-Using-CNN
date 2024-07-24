import cv2
import numpy as np
from keras.models import load_model # type: ignore
from keras.preprocessing.image import img_to_array # type: ignore

# Load the model
try:
    model = load_model("D:/Projects/Emotion Detection System Using CNN/Model.h5")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

emotions = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']

def crop_face(image, x, y, w, h, target_size=(48, 48)):
    try:
        intrested_face = image[y:y + h, x:x + w]
        if len(intrested_face.shape) == 3 and intrested_face.shape[2] == 3:
            intrested_face = cv2.cvtColor(intrested_face, cv2.COLOR_BGR2GRAY)
        intrested_face_resized = cv2.resize(intrested_face, target_size, interpolation=cv2.INTER_AREA)
        return intrested_face_resized
    except Exception as e:
        print(f"Error in crop_face: {e}")
        return None

def detect_emotions(face_image):
    try:
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([face_image]) != 0:
            intrest = face_image.astype('float') / 255.0
            intrest = img_to_array(intrest)
            intrest = np.expand_dims(intrest, axis=0)
            intrest = np.expand_dims(intrest, axis=-1)
            prediction = model.predict(intrest)[0]
            return emotions[prediction.argmax()]
        else:
            print("No face detected")
            return None
    except Exception as e:
        print(f"Error in detect_emotions: {e}")
        return None
