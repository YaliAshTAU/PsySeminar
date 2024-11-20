import argparse
from get_annotations import get_annotations, get_annotation_lists

def get_accuracy(blip_classifications, human_annotations):
    return [blip_classifications[i] == human_annotations[i] for i in range(len(blip_classifications))].count(True) / len(blip_classifications)

def get_precision(blip_classifications, human_annotations):
    true_positives = sum(1 for i in range(len(blip_classifications)) if blip_classifications[i] == 1 and human_annotations[i] == 1)
    false_positives = sum(1 for i in range(len(blip_classifications)) if blip_classifications[i] == 1 and human_annotations[i] == 0)
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

def get_recall(blip_classifications, human_annotations):
    true_positives = sum(1 for i in range(len(blip_classifications)) if blip_classifications[i] == 1 and human_annotations[i] == 1)
    false_negatives = sum(1 for i in range(len(blip_classifications)) if blip_classifications[i] == 0 and human_annotations[i] == 1)
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--annotation_dir", type=str, required=True)
    args.add_argument("--movie_path", type=str, required=True)
    args.add_argument("--pipeline", action='store_true', help="Enable pipeline")
    args.add_argument("--blip", action='store_true', help="Enable blip")
    args = args.parse_args()

    annotation_dir = args.annotation_dir
    movie_path = args.movie_path
    pipeline = args.pipeline
    blip = args.blip

    print("pipeline:", pipeline)
    print("blip:", blip)
    print("Starting test.py")
    social_nonsocial_annotations = get_annotations(annotation_dir, "social_nonsocial")

    blip_classifications, human_annotations = get_annotation_lists(movie_path, social_nonsocial_annotations, blip, pipeline)

    # Print the collected annotations
    print("BLIP Classifications:", blip_classifications)
    print("Human Annotations:", human_annotations)

    # Calculate metrics
    accuracy = get_accuracy(blip_classifications, human_annotations)
    precision = get_precision(blip_classifications, human_annotations)
    recall = get_recall(blip_classifications, human_annotations)
    f1_score = get_f1_score(precision, recall)

    # Print results
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
