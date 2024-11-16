from transformers import BlipProcessor, BlipForQuestionAnswering
from llmUtils import generate_captions
from PIL import Image
import os
import torch

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
        classification = classify_social_interaction(image_path) # Can use also generate_captions(image_path)
        print(f"Image: {image_path}\nClassification: {classification}\n")
        classifications.append(classification)
    
    print(f"Number of 'yes' classifications: {classifications.count('yes')}")
    print(f"Number of 'no' classifications: {classifications.count('no')}")

def get_blip_classifier(pipeline):
    # Load the BLIP VQA model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the questions
    question1 = "Does this image contain social interaction?"
    question2 = "Is there actions or communication between two or more individuals that are directed at and contingent upon each other?"
    
    def classify(image):
        if pipeline:
            # Ask question2 first
            inputs2 = processor(image, question2, return_tensors="pt").to(device)
            out2 = model.generate(**inputs2)
            answer2 = processor.decode(out2[0], skip_special_tokens=True).lower()

            # If question2 is "no", return "no"
            if answer2 == "no":
                return answer2

            # If question2 is "yes", ask question1 (question2 return some false positives but question1 is good at removing them)
            inputs1 = processor(image, question1, return_tensors="pt").to(device)
            out1 = model.generate(**inputs1)
            answer1 = processor.decode(out1[0], skip_special_tokens=True).lower()

            # Return "yes" if both answers are "yes", otherwise "no"
            return "yes" if answer1 == "yes" else "no"
        else: 
            # Process the image and question
            inputs = processor(image, question2, return_tensors="pt").to(device) 
            
            # Get the answer
            out = model.generate(**inputs)
            return processor.decode(out[0], skip_special_tokens=True)
    
    return classify

# classify_images()