from ui import initialize_ui
from camera import start_camera
from image_processing import choose_image

if __name__ == "__main__":
    try:
        initialize_ui(start_camera, choose_image)
    except Exception as e:
        print(f"Error in main.py: {e}")
