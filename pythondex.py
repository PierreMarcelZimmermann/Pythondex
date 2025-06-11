import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import json
import random
from loguru import logger
import subprocess
import cv2

from class_files.image_recognition import ImageRecognition

# Globals and configuration
on_rasp = False
ir = ImageRecognition()
TARGET_SIZE = 256

def set_on_rasp(value):
    """
    Set the mode (Raspberry Pi or other device) and initialize the camera.
    """
    global on_rasp
    on_rasp = value
    logger.info(f"Device type set to: {'Raspberry Pi' if on_rasp else 'Other device'}")

    rasp_button.pack_forget()
    other_button.pack_forget()
    initialize_camera()

def initialize_camera():
    """
    Initialize the camera or show capture options depending on the device.
    """
    if on_rasp:
        logger.info("Initializing Pi Camera...")
    else:
        logger.info("Running in non-Pi mode. No camera initialization needed.")

    capture_button.pack(ipadx=5, ipady=5, expand=True)
    logger.debug("Capture button displayed.")

def take_picture(filename):
    """
    Take a picture using the Pi camera or open a file dialog on non-Pi devices.

    Returns:
        str: Path to the captured or selected image, or None if no image was selected.
    """
    if on_rasp:
        logger.info("Capturing image using Pi Camera...")
        try:
            subprocess.run([
                "libcamera-still",
                "-o", filename,
                "--width", str(TARGET_SIZE),
                "--height", str(TARGET_SIZE),
                "--nopreview",
                "--timeout", "1"
            ], check=True)
            logger.debug(f"Image saved to: {filename}")
            return filename
        except subprocess.CalledProcessError as e:
            logger.error(f"Error capturing image with Pi Camera: {e}")
            return None
    else:
        logger.info("Opening file dialog to select an image...")
        selected_file = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if selected_file:
            logger.info(f"Selected file: {selected_file}")
            return selected_file
        else:
            logger.warning("No file selected by user.")
            return None

def capture_image():
    """
    Capture an image, process it using the recognition model,
    and display both the image and predicted Pokémon information.
    """
    os.makedirs("img", exist_ok=True)
    logger.info("Starting image capture process...")

    image_path = take_picture("img/captured_image.jpg")
    if image_path is None:
        logger.warning("Image capture failed or was canceled.")
        return

    logger.debug(f"Image path to process: {image_path}")
    display_image(image_path)

    try:
        pokemon_id = ir.predict(image_path)
        logger.info(f"Predicted Pokémon ID: {pokemon_id}")
    except Exception as e:
        logger.exception("Prediction failed.")
        display_pokemon_info(("Prediction Error", "Could not identify Pokémon."))
        return

    try:
        pokemon_info = ir.get_pokemon_info(pokemon_id)
        logger.info(f"Retrieved Pokémon info: {pokemon_info}")
    except Exception as e:
        logger.exception("Failed to retrieve Pokémon information.")
        display_pokemon_info(("Unknown Pokémon", "Description unavailable."))
        return

    if isinstance(pokemon_info, tuple) and len(pokemon_info) == 2:
        logger.debug("Displaying Pokémon info on UI.")
        display_pokemon_info(pokemon_info)
    else:
        logger.error(f"Unexpected structure of pokemon_info: {pokemon_info}")
        display_pokemon_info(("Unknown Pokémon", "Description unavailable."))

def display_image(image_path):
    """
    Load and display the image in the GUI.

    Args:
        image_path (str): Path to the image to be displayed.
    """
    try:
        logger.debug(f"Loading image for UI display: {image_path}")
        img = Image.open(image_path)
        img = img.resize((256, 256), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk
        logger.info("Image displayed successfully.")
    except Exception as e:
        logger.exception("Error displaying image.")

def display_pokemon_info(pokemon_info):
    """
    Display Pokémon name and description in the GUI.

    Args:
        pokemon_info (tuple): Tuple containing the Pokémon name and description.
    """
    if pokemon_info:
        name, description = pokemon_info
        logger.debug(f"Updating UI with Pokémon: {name}")
        pokemon_name_label.config(text=f"Pokémon: {name}")
        pokemon_description_label.config(text=f"Description: {description}")
    else:
        logger.warning("No Pokémon info available to display.")
        pokemon_name_label.config(text="Pokémon not found.")
        pokemon_description_label.config(text="Description not available.")

# GUI setup
root = tk.Tk()
root.geometry('600x600')
root.resizable(False, False)
root.title('Pokémon Identifier')
logger.info("Application started. GUI initialized.")

# Buttons for platform selection
rasp_button = ttk.Button(
    root,
    text="I'm using a Raspberry Pi",
    command=lambda: set_on_rasp(True)
)

other_button = ttk.Button(
    root,
    text="I'm using a different device",
    command=lambda: set_on_rasp(False)
)

# Button to trigger image capture
capture_button = ttk.Button(
    root,
    text="Capture Image",
    command=capture_image
)

# Labels for GUI display
image_label = ttk.Label(root)
pokemon_name_label = ttk.Label(root, text="Pokémon: ", wraplength=580)
pokemon_description_label = ttk.Label(root, text="Description: ", wraplength=580)

# Initial button layout
rasp_button.pack(ipadx=5, ipady=5, expand=True)
other_button.pack(ipadx=5, ipady=5, expand=True)
logger.debug("Initial device selection buttons displayed.")

# Remaining layout
image_label.pack()
pokemon_name_label.pack()
pokemon_description_label.pack()
logger.debug("UI components packed.")

# Start GUI loop
logger.info("Entering main loop.")
root.mainloop()
