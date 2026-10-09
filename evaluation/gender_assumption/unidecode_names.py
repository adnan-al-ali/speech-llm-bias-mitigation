import os
import shutil
from unidecode import unidecode

def unidecode_names(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            name, ext = os.path.splitext(filename)
            unidecoded_name = unidecode(name) + ext
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, unidecoded_name)
            shutil.move(src, dst)

if __name__ == "__main__":
    unidecode_names("evaluation/gender_assumption/recordings")