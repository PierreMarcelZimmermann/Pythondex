import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
from loguru import logger

class ImageRecognition:
    model_path = r"resources/pokemon_model.tflite"
    class_ids_path = r"resources/class_ids.json"
    pokemon_data_path = r"resources/pokemon.json"
    img_size = 256

    def __init__(self):
        if not os.path.exists(self.class_ids_path):
            raise FileNotFoundError(f"class_ids.json not found in: {self.class_ids_path}")
        with open(self.class_ids_path, encoding="utf-8") as f:
            self.class_ids = json.load(f)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found in: {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        try:
            with open(self.pokemon_data_path, encoding="utf-8") as f:
                self.pokemons = json.load(f)
                logger.info(f"{len(self.pokemons)} Pokémon loaded from JSON.")
        except Exception as e:
            logger.error(f"Failed to load Pokémon JSON: {e}")
            exit(1)

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found in: {image_path}")
            
        img = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
        img = np.asarray(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_index = int(np.argmax(output[0]))
        return self.class_ids[predicted_index]  # Falls class_ids dict mit string keys

    def get_pokemon_info(self, predicted_index):
        logger.info(f"Searching for Pokémon with ID: {predicted_index}")
        
        for p in self.pokemons:
            pokemon_id = p.get("id")
            logger.debug(f"Checking Pokémon ID: {pokemon_id}")
            
            if int(pokemon_id) == int(predicted_index):
                name = p.get("name", {}).get("english", "Unknown")
                description = p.get("description", "No description available.")
                logger.info(f"Found Pokémon: {name} (ID: {predicted_index})")
                logger.debug(f"Description: {description}")
                return name, description
        
        logger.warning(f"Pokémon with ID {predicted_index} not found.")
        return None, None
