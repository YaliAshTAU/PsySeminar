import argparse
from get_annotations import get_annotations, get_annotation_lists

def get_accuracy(blip_classifications, human_annotations):
    return [blip_classifications[i] == human_annotations[i] for i in range(len(blip_classifications))].count(True) / len(blip_classifications)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--annotation_dir", type=str, required=True)
    args.add_argument("--movie_path", type=str, required=True)
    args.add_argument("--pipeline", type=bool, required=False)
    args = args.parse_args()

    annotation_dir = args.annotation_dir
    movie_path = args.movie_path
    pipeline = args.pipeline

    social_nonosocial_annotations = get_annotations(annotation_dir, "social_nonsocial")

    blip_classifications, human_annotations = get_annotation_lists(movie_path, social_nonosocial_annotations, pipeline)

    # Print the collected annotations
    print("BLIP Classifications:", blip_classifications)
    print("Human Annotations:", human_annotations)

    print('Accuracy:', get_accuracy(blip_classifications, human_annotations))