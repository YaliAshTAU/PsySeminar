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
    question1 = "Does the image contain a person or people?"
    question2 = "Is there an action or communication between two or more individuals that is directed at and contingent upon each other?"
    
    def classify(image):
        if pipeline:
            # Ask question 1
            inputs1 = processor(image, question1, return_tensors="pt").to(device)
            out1 = model.generate(**inputs1)
            answer1 = processor.decode(out1[0], skip_special_tokens=True).lower()

            # If there are no people, return false.
            if answer1 == "no":
                return "no"

            # If there are people, check the social interaction prompt
            # Ask question 2
            inputs2 = processor(image, question2, return_tensors="pt").to(device)
            out2 = model.generate(**inputs2)
            answer2 = processor.decode(out2[0], skip_special_tokens=True).lower()

            return "yes" if answer2 == "yes" else "no"
        else: 
            # Process the image and question
            inputs = processor(image, question1, return_tensors="pt").to(device) 
            
            # Get the answer
            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True).lower()
            print(answer)
            return answer
    
    return classify

# classify_images()