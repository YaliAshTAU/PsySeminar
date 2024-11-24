# Code to play the movie with the annotations
import cv2
import argparse
import scipy.io

def get_annotations(ann_dir, type):
    annotation_file = f"{ann_dir}/{type}.mat"
    mat = scipy.io.loadmat(annotation_file)
    key_name = annotation_file.split("/")[-1].split(".")[0]
    try:
        data = mat[key_name]
    except:
        print ('Error: missing annotation file for ', annotation_file)
    return data


def add_relevant_annotation(all_annotations, count_annotation_frames):
    text = ''
    for key in all_annotations:
        if all_annotations[key][count_annotation_frames] == 1:
            text += f'{key}, '
        else:
            text += f'no {key}, '
    return text


def add_text_to_movie(movie_path, new_path, all_annotations):
    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {movie_path}")
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 24  #for summer-25 taken from https://www.nature.com/articles/s41597-020-00680-2. To get actual fps: fps= cap.get(cv2.CAP_PROP_FPS), for sherlock-24
    out_cap = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print('fps', fps)

    count_seconds = 0
    count_annotation_frames = 25 # delay
    end_second = 3264
    end_frame = end_second * fps
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        frame_count += 1
        if not ret:
            break
        if frame_count > end_frame:
            continue # don't save frames after end_second

        count_seconds += 1 / fps
        # every 3 seconds, increase the annotation frame
        if count_seconds >= 1.5:
            count_annotation_frames += 1
            count_seconds = 0

        text = add_relevant_annotation(all_annotations, count_annotation_frames)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        out_cap.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--annotation_dir", type=str, required=True)
    args.add_argument("--movie_path", type=str, required=True)
    args.add_argument("--new_path", type=str, required=True)
    args = args.parse_args()

    annotation_dir = args.annotation_dir
    movie_path = args.movie_path
    new_path = args.new_path

    faces_annotations = get_annotations(annotation_dir, "face")
    print("Faces annotations loaded:", faces_annotations is not None)
    written_text_annotations = get_annotations(annotation_dir, "written_text")
    print("Written text annotations loaded:", written_text_annotations is not None)
    social_noosocial_annotations = get_annotations(annotation_dir, "social_nonsocial")
    print("Social non-social annotations loaded:", social_noosocial_annotations is not None)
    speaking_annotations = get_annotations(annotation_dir, "speaking")
    print("Speaking annotations loaded:", speaking_annotations is not None)
    # touch_annotations = get_annotations(annotation_dir, "touch")
    all_annotations = {"faces": faces_annotations,
                       "written_text": written_text_annotations,
                       "social": social_noosocial_annotations,
                       "speaking": speaking_annotations}

    add_text_to_movie(movie_path, new_path, all_annotations)
