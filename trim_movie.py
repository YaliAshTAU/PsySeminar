from moviepy.editor import *
clip = VideoFileClip("Sherlock.mp4")
clip = clip.without_audio()
clip = clip.cutout(125, 158)
clip.write_videofile("Sherlock_cut.mp4")