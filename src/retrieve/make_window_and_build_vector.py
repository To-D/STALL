import os
from make_window import make_window
from build_vector import build_vector

if __name__ == "__main__":
    repos = os.listdir("data/repos/cceval/java")
    language = 'java'
    make_window(repos, 20, 2, language)
    build_vector(repos, 20, 2, language)