import io
from PIL import Image
import ollama
from ollama import generate
import os
from io import BytesIO

def image_to_bytes(image_file):
    print('Processing image')
    with Image.open(image_file) as image:
        # Convert the image to bytes
        with io.BytesIO() as output:
            image.save(output, format="png")  # Change the format as needed (e.g., JPEG, PNG)
            image_bytes = output.getvalue()
        return image_bytes

def generate_captions(image_file):
    image_bytes = image_to_bytes(image_file)
    print('Generating captions')
    captions = generate(model='llava:13b', prompt='describe this image in one sentence:', images=[image_bytes])
    return captions["response"]

print(generate_captions('Summer in car.png'))
