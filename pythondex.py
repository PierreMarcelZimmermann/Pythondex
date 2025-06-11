import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import json
import random
from loguru import logger
import subprocess
import cv2

from class_files.image_recognition import ImageRecognition

# variables
on_rasp = False
ir = ImageRecognition()
TARGET_SIZE = 256

# take a picture using the Raspberry Pi camera
def take_picture(filename):
    if on_rasp:
        subprocess.run([
            "libcamera-still",
            "-o", filename,
            "--width", str(TARGET_SIZE),
            "--height", str(TARGET_SIZE),
            "--nopreview",
            "--timeout", "1"
        ])
    else:
        img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(filename)

def show_image(filename, duration=3):
    img = cv2.imread(filename)
    if img is None:
        return

    window_name = "Captured Image"
    cv2.imshow(window_name, img)

    if cv2.waitKey(duration * 1000) == -1:
        pass

    cv2.destroyAllWindows()

    
def display_pokemon_info(pokemon_info):
    if pokemon_info:
        name, description = pokemon_info
        pokemon_name_label.config(text=f"Pokémon: {name}")
        pokemon_description_label.config(text=f"Description: {description}")
    else:
        pokemon_name_label.config(text="Pokémon not found.")
        pokemon_description_label.config(text="Description not available.")

def initialize_camera():
    global picam
    if on_rasp:
        logger.info("Using Pi Camera")

        pass
    else:
        logger.info("Mocking Camera")

    # Show the capture button after initializing the camera
    capture_button.pack(
        ipadx=5,
        ipady=5,
        expand=True
    )

def capture_image():
    # Ensure the img directory exists
    os.makedirs("img", exist_ok=True)
    # Capture image from the camera
    image_path = os.path.join("img", "captured_image.jpg")
    take_picture(image_path)
    logger.info(f"Image captured and saved as {image_path}")
    display_image(image_path)

    # Get the prediction
    pokemon_id = ir.predict(image_path)
    pokemon_info = ir.get_pokemon_info(pokemon_id)
    logger.info(f"ImageRecognition predicted Pokémon: {pokemon_info}")

    # Check if the pokemon_info is a tuple (name, description)
    if isinstance(pokemon_info, tuple) and len(pokemon_info) == 2:
        name, description = pokemon_info
        display_pokemon_info(pokemon_info)
    else:
        logger.error(f"Unexpected pokemon_info structure: {pokemon_info}")
        display_pokemon_info("Unknown Pokémon", "Description unavailable.")


def display_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((256, 256), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        # Create a label to display the image
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        logger.error(f"Error displaying image: {e}")

def set_on_rasp(value):
    global on_rasp
    on_rasp = value
    rasp_button.pack_forget()
    other_button.pack_forget()
    initialize_camera()

root = tk.Tk()
root.geometry('600x600')
root.resizable(False, False)
root.title('Button Demo')

rasp_button = ttk.Button(
    root,
    text='I\'m using a Raspberry Pi',
    command=lambda: set_on_rasp(True)
)

other_button = ttk.Button(
    root,
    text='I\'m using a different device',
    command=lambda: set_on_rasp(False)
)

capture_button = ttk.Button(
    root,
    text='Capture Image',
    command=capture_image
)

# label to display the captured image
image_label = ttk.Label(root)

# label to display the predicted Pokémon name
pokemon_name_label = ttk.Label(root, text="Pokémon: ", wraplength=580)
pokemon_description_label = ttk.Label(root, text="Description: ", wraplength=580)

rasp_button.pack(
    ipadx=5,
    ipady=5,
    expand=True
)

other_button.pack(
    ipadx=5,
    ipady=5,
    expand=True
)

image_label.pack()
pokemon_name_label.pack()
pokemon_description_label.pack()


root.mainloop()
