from PIL import Image
from ollama import generate
from io import BytesIO
from transformers import BlipProcessor, BlipForQuestionAnswering
import time
import base64
import requests

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

def query_ollama(model="llava:13b", prompt="", images=None, url="https://b801-34-16-230-149.ngrok-free.app/api/generate", stream=False):
    """
    Sends a prompt and image to the Ollama server and retrieves the response.
    """
    if images is None:
        images = []
    
    headers = {"Content-Type": "application/json"}
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

def classify_using_llava(image_file, prompt='describe this photo in 2 sentences', llama = False): #Are there people in the image? If yes, is there social interaction between the people? Answer only Yes or No
    """
    Classifies an image using the LLAVA model and answers the given prompt.
    """
    image_base64 = image_to_base64(image_file)
    # print('Generating caption')
    # Start timing
    start_time = time.perf_counter()
    if llama:
        captions = query_ollama(model='llava:13b', prompt="Describe this image in 2 sentences", images=[image_base64]) # generate(model='llava:13b', prompt=prompt, images=[image_base64])
    else:
        captions = query_ollama(model='llava:13b', prompt=prompt, images=[image_base64])
    # print('caption:', captions["response"])
    answer = captions["response"].lower()
    if llama:
        # Query for llama model
        llama_prompt = "For this image caption:" + answer + "," + prompt
        captions = query_ollama(model='llama3', prompt=llama_prompt, url='https://f3b1-35-187-248-59.ngrok-free.app/api/generate') # generate(model='llava:13b', prompt=prompt, images=[image_base64])
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

