import scenedetect as sd
import cv2
from PIL import Image
import torch
import uuid
import os
import matplotlib.pyplot as plt
from collections import Counter


class Scene:
    def __init__(self, video_path, scene_idx, scene) -> None:
        self.video_path = video_path
        self.video = sd.open_video(video_path)
        sm = sd.SceneManager()
        sm.add_detector(sd.ContentDetector(threshold=27.0))
        self.scene_idx = scene_idx
        self.scene = scene
        self.frames = None
        self.cap  = cv2.VideoCapture(video_path)
        self.scene_clip_embeddings = []

    def save_tensor(self, t, directory='saved_tensors'):
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        filename = str(uuid.uuid4()) + '.pt'    # Generate a unique filename
        path = os.path.join(directory, filename) # Construct the full path
        torch.save(t, path)
        return path
    
    def set_frames_for_scene(self):
        no_of_samples = 5 # number of samples per scene
        scene_length = abs(self.scene[0].frame_num - self.scene[1].frame_num)
        every_n = round(scene_length/no_of_samples)
        samples = [(every_n * n) + self.scene[0].frame_num for n in range(no_of_samples)]     
        self.frames = samples

    def clip_embeddings(self, image, clip_processor):
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        input_tokens = {
            k: v for k, v in inputs.items()
        }
        return input_tokens['pixel_values']
    
    def get_prediction(self, clip_processor, clip_model, classes):
        frame_labels = []
        inputs = clip_processor(text=classes, return_tensors="pt", padding=True)
        for tensor in self.scene_clip_embeddings:
            #tensor = torch.load(tensor)
            inputs['pixel_values'] = tensor   
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            highest_score_idx = probs.argmax(dim=1).item()  # Get the index of the highest score
            frame_labels.append(classes[highest_score_idx])  # Append the corresponding class label
        
        # for i in range(len(self.scene_clip_embeddings)):
            # plt.barh(range(len(probs[0].detach().numpy())),probs[i].detach().numpy(), tick_label=classes)
            # plt.xlim(0,1.0)
            # plt.subplots_adjust(left=0.1,
            #                     bottom=0.1,
            #                     right=0.9,
            #                     top=0.9,
            #                     wspace=0.2,
            #                     hspace=0.8)
            # plt.show()
        most_common_label = Counter(frame_labels).most_common(1)[0][0]
        print(f"Most common label for scene {self.scene_idx}: {most_common_label}")
        print(frame_labels)

    def embed_scene(self, clip_processor):
        print("embedding scene #", self.scene_idx)
        pixel_tensors = [] # holds all of the clip embeddings for each of the samples
        for frame_sample in self.frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_sample)
            ret, frame = self.cap.read()
            if not ret:
                print('failed to read', ret, frame_sample, self.scene_idx, frame)
                return
            pil_image = Image.fromarray(frame)
            clip_pixel_values = self.clip_embeddings(pil_image, clip_processor)
            self.scene_clip_embeddings.append(clip_pixel_values)
            # pixel_tensors.append(clip_pixel_values)
        # avg_tensor = torch.mean(torch.stack(pixel_tensors), dim=0)
        # self.scene_clip_embeddings.append(self.save_tensor(avg_tensor))

    