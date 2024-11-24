import base64
from PIL import Image
from io import BytesIO
import time
import base64
import requests
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

def image_to_bytes(image_input):
    """
    Converts an image to bytes for transmission to the API.
    """
    # print('Processing image')
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input)
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return buffered.getvalue()

def image_to_base64(image_input):
    """
    Converts an image to a base64-encoded string.
    """
    image_bytes = image_to_bytes(image_input)
    return base64.b64encode(image_bytes).decode('utf-8')

def query_ollama(model="llava:13b", prompt="", images=None, url="https://7443-35-247-153-127.ngrok-free.app/api/generate", stream=False):
    """
    Sends a prompt and image to the Ollama server and retrieves the response.
    """
    if images is None:
        images = []
    
    headers = {"Content-Type": "application/json"} # "Authorization": "Bearer 2QYewyTfJQyrqq25GUOdzXgyUHp_51c8WpyiTvBFCtnmCrnss"
    if model == "llava:13b": 
        payload = {
            "model": model,
            "prompt": prompt,
            "images": images,
            "stream": stream
        }

    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()  # Return the full response JSON
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except ValueError:
        return "Error: Unable to parse JSON response."

def classify_using_llava(image_file, prompt='describe this image shortly', llama = False): #Are there people in the image? If yes, is there social interaction between the people? Answer only Yes or No
    """
    Classifies an image using the LLAVA model and answers the given prompt.
    """
    image_base64 = image_to_base64(image_file)
    # print('Generating caption')
    # Start timing
    start_time = time.perf_counter()
    if llama:
        captions = query_ollama(model='llava:13b', prompt="Describe this image shortly", images=[image_base64]) # generate(model='llava:13b', prompt=prompt, images=[image_base64])
    else:
        captions = query_ollama(model='llava:13b', prompt=prompt, images=[image_base64])
    print('caption:', captions)
    answer = captions["response"].lower()
    if llama:
        # Query for llama model
        llama_prompt = "For this image caption:" + answer + "," + prompt
        captions = query_ollama(model='llama3', prompt=llama_prompt) # generate(model='llava:13b', prompt=prompt, images=[image_base64])
        # print('llama response:', captions)
        # End query for llama model
    # End timing
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # print(f'Caption generated in {elapsed_time:.2f} seconds')
    answer = captions["response"].lower()
    # print('Answer before process:', answer)
    if "yes" in answer:
        answer = "yes"
    else:
        answer = "no"
    # print('Answer:', answer)
    return answer

def get_blip_classifier():
    # Load the BLIP VQA model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model


def classify_using_blip(image, processor, model, pipeline=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question1 = "Are there any people, human beings or faces in the image? answer only Yes or No"
    question2 = "Is there direct social interaction between people in the image? Answer only Yes or No" # "Is there an action or communication between two or more individuals that is directed at and contingent upon each other? answer only Yes or No"
    if pipeline:
        # Ask question 1
        inputs1 = processor(image, question1, return_tensors="pt").to(device)
        out1 = model.generate(**inputs1)
        answer1 = processor.decode(out1[0], skip_special_tokens=True).lower()
        # print("Answer 1:", answer1)
        #If answer is not "yes" or "no", throw an error

        # If there are no people, return false.
        if "yes" not in answer1:
            print("Answer 1:", answer1)
            return "no"

        # If there are people, check the social interaction prompt
        # Ask question 2
        inputs2 = processor(image, question2, return_tensors="pt").to(device)
        out2 = model.generate(**inputs2)
        answer2 = processor.decode(out2[0], skip_special_tokens=True).lower()
        print("Answer 2:", answer2)

        return "yes" if "yes" in answer2 else "no"
    
    else: 
        # Process the image and question
        inputs = processor(image, question2, return_tensors="pt").to(device) 
        # Get the answer
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True).lower()
        return answer
        
    
