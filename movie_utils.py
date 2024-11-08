import cv2
from moviepy.editor import VideoFileClip

def cut_movie(movie_path, new_path, start_frame, end_frame):
    cap = cv2.VideoCapture(movie_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_cap = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count < start_frame:
            continue
        if count > end_frame:
            break
        out_cap.write(frame)

    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()


def save_with_sound(movie_path, output_video_path):
    start_time = 10
    end_time = 3000
    output_fps = 25

    # Load the video file
    video = VideoFileClip(movie_path).subclip(start_time, end_time)

    # Write the result to a new file, including audio, with a different frame rate
    video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=output_fps)

    print("Segment with text and different frame rate saved successfully!")