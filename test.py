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
    # args = argparse.ArgumentParser()
    # args.add_argument("--annotation_dir", type=str, required=True)
    # args.add_argument("--movie_path", type=str, required=True)
    # args.add_argument("--movie_name", type=str, required=True) # 'summer' or 'sherlock'
    # args.add_argument("--pipeline", action='store_true', help="Enable pipeline")
    # args.add_argument("--blip", action='store_true', help="Enable blip")
    # args = args.parse_args()

    # annotation_dir = args.annotation_dir
    # movie_path = args.movie_path
    # movie_name = args.movie_name
    # pipeline = args.pipeline
    # blip = args.blip

    # print("pipeline:", pipeline)
    # print("blip:", blip)
    # print("Starting test.py")

    annotation_dirs = ['Summer/', 'Sherlock/']
    movie_paths = ['Summer.mp4', 'Sherlock.mp4']
    movie_names = ['summer', 'sherlock']
    llamas = [True, False]
    pipeline = False
    blip = False

    prompts = ['Analyze the image and determine if there is social interaction present. Social interaction is defined as an action or communication between two or more individuals that is directed at and contingent upon each other. Respond with "yes" if there is social interaction, or "no" if there is not. Do not provide any additional text.',
                'In this image, Is there actions or communication between two or more individuals that are directed at and contingent upon eachother?',
                # 'Are there people in the image? If yes, is there social interaction between the people? Answer only Yes or No',
                # 'In the image, is there direct social interaction between people in the image? Answer only Yes or No'
                ]

    results = []

    for i in range(2):
        annotation_dir = annotation_dirs[i]
        movie_path = movie_paths[i]
        movie_name = movie_names[i]
        # for llama in llamas:
        for prompt in prompts:
            social_nonsocial_annotations = get_annotations(annotation_dir, "social_nonsocial")
            # print("Social Nonsocial Annotations:", social_nonsocial_annotations.tolist())

            model_classifications, human_annotations = get_annotation_lists(movie_path, social_nonsocial_annotations, blip, pipeline, movie_name, prompt, llama=False)

            # Print the collected annotations
            print("Model Classifications:", model_classifications)
            print("Human Annotations:", human_annotations)

            # Calculate metrics
            accuracy = get_accuracy(model_classifications, human_annotations)
            precision = get_precision(model_classifications, human_annotations)
            recall = get_recall(model_classifications, human_annotations)
            f1_score = get_f1_score(precision, recall)

            # Print results
            print('Accuracy:', accuracy)
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1 Score:', f1_score)

            results.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })

    print(results)
