import scenedetect as sd
from Scene import Scene
import cv2
from transformers import CLIPProcessor, CLIPModel

class Video:
    def set_frames(self):
        for scene in self.scenes:
            scene.set_frames_for_scene()

    def embed_scenes(self):
        for scene in self.scenes:
            scene.embed_scene(self.clip_processor)
    
    def predict(self):
        for scene in self.scenes:
            scene.get_prediction(self.clip_processor, self.model, self.classes)

    def __init__(self, video_path, classes) -> None:
        self.video_path = video_path
        self.video = sd.open_video(video_path)
        self.classes = classes
        sm = sd.SceneManager()
        sm.add_detector(sd.ContentDetector(threshold=27.0))
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print("detecting scenes")
        sm.detect_scenes(self.video)
        print("scenes detected. getting list")
        scene_list = sm.get_scene_list()
        print('got list')
        print(len(scene_list))
        self.scenes = [Scene(video_path, scene_idx, scene_list[scene_idx]) for scene_idx in range(len(scene_list))]
        print("scenes created")
        self.set_frames()
        print("frames set")
        self.embed_scenes()
        print("Initiated")
        self.predict()

    def print_scenes(self):
        for i, scene in enumerate(self.scenes, start=1):
            print(f"Scene {i}: {scene.scene}")

    def print_frames(self):
        for scene in self.scenes:
            print(f"Scene {scene.scene_idx}: {scene.frames}")



    

        


