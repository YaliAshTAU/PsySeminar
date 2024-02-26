import scenedetect as sd
import cv2
from transformers import CLIPProcessor
from PIL import Image
import torch
import uuid
import os


class Scene:
    def __init__(self, video_path, scene_idx, scene) -> None:
        self.video_path = video_path
        self.video = sd.open_video(video_path)
        sm = sd.SceneManager()
        sm.add_detector(sd.ContentDetector(threshold=27.0))
        self.scene_idx = scene_idx
        self.scene = scene
        self.frames = None
        self.clip_processor = clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.cap  = cv2.VideoCapture(video_path)
        self.scene_clip_embeddings = []

    def save_tensor(self, t, directory='saved_tensors'):
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        filename = str(uuid.uuid4()) + '.pt'    # Generate a unique filename
        path = os.path.join(directory, filename) # Construct the full path
        torch.save(t, path)
        return path
    
    def set_frames_for_scene(self):
        #cap = cv2.VideoCapture(self.video_path)
        no_of_samples = 5 # number of samples per scene
        scene_length = abs(self.scene[0].frame_num - self.scene[1].frame_num)
        every_n = round(scene_length/no_of_samples)
        samples = [(every_n * n) + self.scene[0].frame_num for n in range(no_of_samples)]     
        self.frames = samples

    def clip_embeddings(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        input_tokens = {
            k: v for k, v in inputs.items()
        }
        return input_tokens['pixel_values']
    

    def embedd_scene(self):
        pixel_tensors = [] # holds all of the clip embeddings for each of the samples
        for frame_sample in self.frames:
            self.cap.set(1, frame_sample)
            ret, frame = self.cap.read()
            if not ret:
                print('failed to read', ret, frame_sample, self.scene_idx, frame)
                break
            pil_image = Image.fromarray(frame)
            clip_pixel_values = self.clip_embeddings(pil_image)
            pixel_tensors.append(clip_pixel_values)
            avg_tensor = torch.mean(torch.stack(pixel_tensors), dim=0)
            print("Tensor shape: ", avg_tensor.shape)
            self.scene_clip_embeddings.append(self.save_tensor(avg_tensor))

    