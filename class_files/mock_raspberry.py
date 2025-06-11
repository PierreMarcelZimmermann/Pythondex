from PIL import Image
import os
from loguru import logger
import random

class MockPicamera2:
    """
    Mock class for picamera2.PiCamera2.
    """
    def create_preview_configuration(self):
        logger.info("Mock: create_preview_configuration()")
        return "mock_config"

    def configure(self, config):
        logger.info(f"Mock: configure({config})")

    def start_preview(self, preview_type):
        logger.info(f"Mock: start_preview({preview_type})")

    def start(self):
        logger.info("Mock: start()")

    def capture_file(self, filename):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (640, 480), color=color)
        img.save(filename, format='JPEG')
        print(f"Mock image saved as {filename}")

    def stop_preview(self):
        logger.info("Mock: stop_preview()")


class Preview:
    """
    Mock preview type.
    """
    QTGL = "mock_preview"
