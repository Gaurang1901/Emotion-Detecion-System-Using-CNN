import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

def initialize_ui(start_camera, choose_image):
    global image_label, emotion_label
    
    try:
        root = tk.Tk()
        root.title("Emotion Detection System")
        root.geometry("1020x720+40+40")
        root.configure(bg='#2E2E2E')

        title_label = tk.Label(root, text="Emotion Detection System", font=("Times", 25, "bold"), bg='#2E2E2E', fg='white')
        title_label.pack(pady=20)

        camera_button = tk.Button(root, text="Detect Emotions via Camera", font=("Helvetica", "15", "bold"), command=start_camera, bg='#2E2E2E', fg='white')
        camera_button.pack(pady=20)

        image_button = tk.Button(root, text="Detect Emotions via Local Image", command=choose_image, bg='#2E2E2E', fg='white', font=("Helvetica", "15", "bold"))
        image_button.pack(pady=20)

        image_label = tk.Label(root, bg='#2E2E2E')
        image_label.pack(pady=10)

        emotion_label = tk.Label(root, text="", font=("Helvetica", 16), bg='#2E2E2E', fg='white')
        emotion_label.pack(pady=10)

        root.mainloop()
    except Exception as e:
        print(f"Error in initialize_ui: {e}")

def display_image(image, emotion):
    try:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (300, 300))
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.config(image=imgtk)
        image_label.image = imgtk
        emotion_label.config(text=f'Detected Emotion: {emotion}')
    except Exception as e:
        print(f"Error in display_image: {e}")
