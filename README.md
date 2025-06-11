# Pokémon Image Recognition
This is a simple Python app that uses a trained TensorFlow Lite model to recognize Pokémon from images. It works on both Raspberry Pi (with a camera) and other devices (with image file selection).
## Features
- Image classification using a TFLite model
- Pokémon name and description lookup from JSON files
- Works with Raspberry Pi camera or image file dialog
- GUI built with tkinter
- Image display using Pillow and OpenCV
- Logging with loguru
## Requirements
See `environment.yml` for all dependencies. Main ones:

- Python 3.10
- tensorflow
- numpy
- pillow
- opencv-python
- loguru
- tkinter (included with Python)
## Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/pokemon-image-recognition.git
cd pokemon-image-recognition
```
2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate pokemon-app
```
## How to Run
Run the app with:
```bash
python pythondex.py
```
Choose your platform:
- Raspberry Pi: captures an image using the Pi camera
- Other devices: lets you select an image file

After capturing or selecting an image, the app shows the image and displays the predicted Pokémon name and description.

## File Structure

- `main.py`: main app and GUI
- `class_files/image_recognition.py`: image classification logic
- `resources/pokemon_model.tflite`: the TFLite model
- `resources/class_ids.json`: mapping from model output to Pokémon ID
- `resources/pokemon.json`: Pokémon name and description data
- `img/`: stores captured or selected images

## Notes

- Raspberry Pi mode requires `libcamera-still` to be installed.
- The model expects images of size 256x256 (handled automatically).
- Logging is done to the console using loguru.

## License

MIT License. For educational and demonstration purposes only.

Pokémon is a trademark of Nintendo and Game Freak. This project is not affiliated with them.