import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
from loguru import logger

class ImageRecognition:
    """
    A class to handle image recognition of Pokémon using a TensorFlow Lite model.
    """
    
    model_path = r"resources/pokemon_model.tflite"
    class_ids_path = r"resources/class_ids.json"
    pokemon_data_path = r"resources/pokemon.json"
    img_size = 256

    def __init__(self):
        """
        Initialize the ImageRecognition model by loading class IDs,
        the TFLite model, and Pokémon information from JSON files.
        """
        logger.info("Initializing ImageRecognition...")

        self._load_class_ids()
        self._load_model()
        self._load_pokemon_data()

    def _load_class_ids(self):
        """
        Load class ID mappings from JSON file.
        """
        if not os.path.exists(self.class_ids_path):
            logger.critical(f"class_ids.json not found at: {self.class_ids_path}")
            raise FileNotFoundError(f"class_ids.json not found in: {self.class_ids_path}")
        
        with open(self.class_ids_path, encoding="utf-8") as f:
            self.class_ids = json.load(f)
            logger.info(f"{len(self.class_ids)} class IDs loaded.")

    def _load_model(self):
        """
        Load the TensorFlow Lite model and prepare it for inference.
        """
        if not os.path.exists(self.model_path):
            logger.critical(f"Model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Model not found in: {self.model_path}")
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info("TFLite model loaded and tensors allocated.")
        except Exception as e:
            logger.exception(f"Failed to initialize TFLite interpreter: {e}")
            raise

    def _load_pokemon_data(self):
        """
        Load Pokémon metadata (name, description) from JSON file.
        """
        if not os.path.exists(self.pokemon_data_path):
            logger.critical(f"Pokémon data file not found at: {self.pokemon_data_path}")
            raise FileNotFoundError(f"pokemon.json not found in: {self.pokemon_data_path}")
        
        try:
            with open(self.pokemon_data_path, encoding="utf-8") as f:
                self.pokemons = json.load(f)
                logger.info(f"{len(self.pokemons)} Pokémon loaded from JSON.")
        except Exception as e:
            logger.exception("Failed to load Pokémon data.")
            raise

    def predict(self, image_path):
        """
        Predict the Pokémon class from a given image.

        Args:
            image_path (str): Path to the image to be classified.

        Returns:
            str: Predicted class ID as found in class_ids.json.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found in: {image_path}")

        logger.debug(f"Processing image: {image_path}")
        try:
            img = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
            img_array = np.asarray(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            logger.debug("Running inference...")
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_index = int(np.argmax(output[0]))

            logger.info(f"Prediction complete. Class ID: {predicted_index}")
            return self.class_ids[predicted_index]
        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            raise

    def get_pokemon_info(self, predicted_index):
        """
        Retrieve the name and description of the predicted Pokémon.

        Args:
            predicted_index (int or str): The predicted class ID.

        Returns:
            tuple: (Pokémon name (str), description (str)), or (None, None) if not found.
        """
        logger.info(f"Looking up Pokémon info for ID: {predicted_index}")
        
        try:
            for pokemon in self.pokemons:
                pokemon_id = pokemon.get("id")
                logger.debug(f"Checking Pokémon ID: {pokemon_id}")

                if int(pokemon_id) == int(predicted_index):
                    name = pokemon.get("name", {}).get("english", "Unknown")
                    description = pokemon.get("description", "No description available.")
                    logger.info(f"Pokémon found: {name}")
                    return name, description

            logger.warning(f"Pokémon with ID {predicted_index} not found in data.")
            return None, None
        except Exception as e:
            logger.exception(f"Failed to retrieve Pokémon info: {e}")
            return None, None
