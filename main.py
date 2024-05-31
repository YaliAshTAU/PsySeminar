import sys
import uuid
from Video import Video

if __name__ == "__main__":
    the_office = Video("You're so white - The Office US.mp4", classes=['hugging', 'talking', 'fighting', 'laughing', 'flirting'])
    the_office.print_scenes()
    the_office.print_frames()
