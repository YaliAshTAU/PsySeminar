import scenedetect as sd
from Scene import Scene
import cv2

class Video:
    def set_frames(self):
        for scene in self.scenes:
            scene.set_frames_for_scene()

    def embedd_scenes(self):
        for scene in self.scenes:
            scene.embedd_scene()

    def __init__(self, video_path) -> None:
        self.video_path = video_path
        self.video = sd.open_video(video_path)
        sm = sd.SceneManager()
        sm.add_detector(sd.ContentDetector(threshold=27.0))
        print("detecting scenes")
        sm.detect_scenes(self.video)
        print("scenes detected. getting list")
        scene_list = sm.get_scene_list()
        self.scenes = [Scene(video_path, scene_idx, scene_list[scene_idx]) for scene_idx in range(len(scene_list))]
        self.set_frames()
        print("frames set")
        self.embedd_scenes()
        print("Initiated")

    

    def print_scenes(self):
        for i, scene in enumerate(self.scenes, start=1):
            print(f"Scene {i}: {scene.scene}")

    def print_frames(self):
        for scene in self.scenes:
            print(f"Scene {scene.scene_idx}: {scene.frames}")



    

        


