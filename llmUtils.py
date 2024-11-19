import io
from PIL import Image
import ollama
from ollama import generate
import os
from io import BytesIO
from transformers import BlipProcessor, BlipForQuestionAnswering

def image_to_bytes(image_file):
    print('Processing image')
    with Image.open(image_file) as image:
        # Convert the image to bytes
        with io.BytesIO() as output:
            image.save(output, format="png")  # Change the format as needed (e.g., JPEG, PNG)
            image_bytes = output.getvalue()
        return image_bytes

def classify_using_llava(image_file, prompt='Are there people in the image? If yes, is there social interaction between the people? Answer only Yes or No'):
    image_bytes = image_to_bytes(image_file)
    print('Generating captions')
    captions = generate(model='llava:13b', prompt=prompt, images=[image_bytes])
    return captions["response"]

def classify_using_blip(image_path):
    # Load the BLIP VQA model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    # Open the image file
    image = Image.open(image_path)
    
    # Define the question
    # question = "Does this image contain social interaction?"
    question = "Is there actions or communication between two or more individuals that are directed at and contingent upon eachother?"
    
    # Process the image and question
    inputs = processor(image, question, return_tensors="pt")
    
    # Get the answer
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return answer


print(classify_using_llava('Pics/Summer in car.png'))
