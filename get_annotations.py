import cv2
import scipy.io
from PIL import Image
from blip_vqa import get_blip_classifier
from llmUtils import classify_using_llava

def get_annotations(ann_dir, type):
    print('getting annotations')
    annotation_file = f"{ann_dir}/{type}.mat"
    mat = scipy.io.loadmat(annotation_file)
    key_name = annotation_file.split("/")[-1].split(".")[0]
    try:
        data = mat[key_name]
    except:
        print ('Error: missing annotation file for ', annotation_file)
    return data

def get_annotation_by_index(annotations_list, index):
    value = int(annotations_list[index][0])
    return value > 0

def get_annotation_lists(movie_path, annotations, blip, pipeline, movie, prompt, llama):
    print('getting annotations list')
    print('number of annotations:', len(annotations))
    if blip:
        get_blip_classification = get_blip_classifier(pipeline)
    cap = cv2.VideoCapture(movie_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25 if movie == "summer" else 24  #taken from https://www.nature.com/articles/s41597-020-00680-2. To get actual fps: fps= cap.get(cv2.CAP_PROP_FPS)

    count_seconds = 0
    start_second = 40 if movie == "summer" else 0 # skip start credits
    count_annotation_frames = 25 if movie == "sherlock" else 0
    start_frame = start_second * fps
    frame_count = 0

    classifications = []
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
        if count_seconds >= (3 if movie == "summer" else 1.5):
            count_annotation_frames += 1
            count_seconds = 0
            pil_image = Image.fromarray(frame)
            if blip:
                classification = True if get_blip_classification(pil_image) == 'yes' else False
            else:
                classification = True if classify_using_llava(pil_image, prompt, llama) == 'yes' else False
            classifications.append(classification)

            human_annotation = get_annotation_by_index(annotations, count_annotation_frames - 1)
            human_annotations.append(human_annotation)
            print('human_annotation: ', human_annotation, ' classification: ', classification)

            is_correct = human_annotation == classification
            # if human_annotation and not classification: # save false negatives
            #     pil_image.save(f'./false_negatives_are_there_people/{count_annotation_frames}-{human_annotation}-{classification}.jpg')
            print('annotation #:',count_annotation_frames, is_correct)

    cap.release()
    return classifications, human_annotations

