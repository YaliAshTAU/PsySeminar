import cv2
import argparse
import scipy.io
from PIL import Image
from blip_vqa import get_blip_classifier

def get_annotations(ann_dir, type):
    annotation_file = f"{ann_dir}/{type}.mat"
    mat = scipy.io.loadmat(annotation_file)
    key_name = annotation_file.split("/")[-1].split(".")[0]
    try:
        data = mat[key_name]
    except:
        print ('Error: missing annotation file for ', annotation_file)
    return data

def get_human_annotation(ann_dir, type):
    annotation_file = f"{ann_dir}/{type}.mat"
    mat = scipy.io.loadmat(annotation_file)
    key_name = annotation_file.split("/")[-1].split(".")[0]
    try:
        data = mat[key_name]
    except:
        print ('Error: missing annotation file for ', annotation_file)
    return data


def get_annotation_by_index(annotations_list, index):
    return annotations_list[index][0] != 0

def get_pre_loaded_blip_classification(image):
    return 


def get_annotation_lists(movie_path, annotations):
    print('getting annotations list')
    print('number of annotations:', len(annotations))
    get_blip_classification = get_blip_classifier()
    cap = cv2.VideoCapture(movie_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25  #taken from https://www.nature.com/articles/s41597-020-00680-2. To get actual fps: fps= cap.get(cv2.CAP_PROP_FPS)

    count_seconds = 0
    start_second = 40 # skip start credits
    count_annotation_frames = 0
    start_frame = start_second * fps
    frame_count = 0

    blip_classifications = []
    human_annotations = []

    while cap.isOpened() and count_annotation_frames < len(annotations):
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count < start_frame:
            continue # don't save frames before start_second

        
        count_seconds += 1 / fps
        # every 3 seconds, increase the annotation frame
        if count_seconds >= 3:
            count_annotation_frames += 1
            count_seconds = 0
            pil_image = Image.fromarray(frame)
            blip_classification = True if get_blip_classification(pil_image) == 'yes' else False
            blip_classifications.append(blip_classification)

            human_annotation = get_annotation_by_index(annotations, count_annotation_frames - 1)
            human_annotations.append(human_annotation)

            is_correct = human_annotation == blip_classification
            if human_annotation and not blip_classification: # save false negatives
                pil_image.save(f'./false_negatives_leyla_prompt/{count_annotation_frames}-{human_annotation}-{blip_classification}.jpg')
            print('annotation #:',count_annotation_frames, is_correct)

    cap.release()
    return blip_classifications, human_annotations

def get_accuracy(blip_classifications, human_annotations):
    return [blip_classifications[i] == human_annotations[i] for i in range(len(blip_classifications))].count(True) / len(blip_classifications)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--annotation_dir", type=str, required=True)
    args.add_argument("--movie_path", type=str, required=True)
    args = args.parse_args()

    annotation_dir = args.annotation_dir
    movie_path = args.movie_path

    social_nonosocial_annotations = get_annotations(annotation_dir, "social_nonsocial")

    blip_classifications, human_annotations = get_annotation_lists(movie_path, social_nonosocial_annotations)

    # Print the collected annotations
    print("BLIP Classifications:", blip_classifications)
    print("Human Annotations:", human_annotations)

    print('Accuracy:', get_accuracy(blip_classifications, human_annotations))