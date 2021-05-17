from extract_emotion_labels import main as extract_emotion_labels
from extract_audio_features import main as extract_audio_features
from build_audio_vectors import main as build_audio_vectors
from prepare_data import main as prepare_data
from config import data_config

def main():
    extract_emotion_labels()
    build_audio_vectors()
    extract_audio_features()
    prepare_data()

if __name__ == '__main__':
    main()