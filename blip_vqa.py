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
    question = "Does this image contain social interaction?"
    
    # Process the image and question
    inputs = processor(image, question, return_tensors="pt")
    
    # Get the answer
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return answer

# # Path to the images folder
# images_folder = 'images'

# # Generate paths for images labeled 1-10
# image_paths = [os.path.join(images_folder, f"{i}.png") for i in range(1, 11)]

# # Classify each image
# for image_path in image_paths:
#     classification = classify_social_interaction(image_path)
#     print(f"Image: {image_path}\nClassification: {classification}\n")

def get_blip_classifier():
    # Load the BLIP VQA model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # Define the question
    question = "Does this image contain social interaction?"
    
    def classify(image):
        # Process the image and question
        inputs = processor(image, question, return_tensors="pt")
        
        # Get the answer
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    
    return classify
