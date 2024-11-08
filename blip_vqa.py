from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import os

def classify_social_interaction(image_path):
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

def classify_images():
    # Path to the images folder
    images_folder = './test'

    # Get a list of all files in the images folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    classifications = []
    # Classify each image
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        classification = classify_social_interaction(image_path)
        print(f"Image: {image_path}\nClassification: {classification}\n")
        classifications.append(classification)
    
    print(f"Number of 'yes' classifications: {classifications.count('yes')}")
    print(f"Number of 'no' classifications: {classifications.count('no')}")

def get_blip_classifier():
    # Load the BLIP VQA model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # Define the question
    # question = "Does this image contain social interaction?"
    question = "Is there actions or communication between two or more individuals that are directed at and contingent upon eachother?"
    
    def classify(image):
        # Process the image and question
        inputs = processor(image, question, return_tensors="pt")
        
        # Get the answer
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    
    return classify

# classify_images()