import cv2
import scipy.io
import argparse
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import scenedetect as sd
from scenedetect import ContentDetector

def get_annotations(ann_dir, type):
    annotation_file = f"{ann_dir}/{type}.mat"
    mat = scipy.io.loadmat(annotation_file)
    key_name = annotation_file.split("/")[-1].split(".")[0]
    try:
        data = mat[key_name]
    except:
        print(f'Error: missing annotation file for {annotation_file}')
    return data

def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

def classify_scene(processor, model, image):
    print('classifying scene')
    question = "What is happening in this scene? Choose from: social interaction, written text, face visible, speaking, touching."
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def detect_scenes(video_path):
    print('in detect_scenes')
    video = sd.open_video(video_path)
    scene_manager = sd.SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()

def process_video(video_path, annotation_dir, processor, model):
    print('processing video')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_annotations = {
        "faces": get_annotations(annotation_dir, "face"),
        "written_text": get_annotations(annotation_dir, "written_text"),
        "social": get_annotations(annotation_dir, "social_nonsocial"),
        "speaking": get_annotations(annotation_dir, "speaking"),
        "touch": get_annotations(annotation_dir, "touch")
    }
    
    scenes = detect_scenes(video_path)
    results = []

    for scene in scenes:
        print(scene)
        start_frame = scene[0].frame_num
        end_frame = scene[1].frame_num
        mid_frame = (start_frame + end_frame) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        blip_classification = classify_scene(processor, model, image)

        annotation_frame = int(mid_frame / (fps * 3))  # Convert to 3-second intervals
        actual_annotations = {k: v[0][annotation_frame] for k, v in all_annotations.items()}

        results.append({
            "scene_start": start_frame,
            "scene_end": end_frame,
            "blip_classification": blip_classification,
            "actual_annotations": actual_annotations
        })

    cap.release()
    return results

def compare_results(results):
    for result in results:
        print(f"Scene: {result['scene_start']} - {result['scene_end']}")
        print(f"BLIP Classification: {result['blip_classification']}")
        print("Actual Annotations:")
        for k, v in result['actual_annotations'].items():
            print(f"  {k}: {'Yes' if v == 1 else 'No'}")
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--movie_path", type=str, required=True)
    args = parser.parse_args()

    processor, model = load_blip_model()
    print('model loaded')
    results = process_video(args.movie_path, args.annotation_dir, processor, model)
    compare_results(results)