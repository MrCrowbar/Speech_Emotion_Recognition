"""
This script preprocesses data and prepares data to be actually used in training
"""
import re
import os
import pickle
import unicodedata
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from config import data_config

# Prepare audio files
def prepare_audio_files():
    path = os.path.join(data_config["pre_processed"], 'audio_features.csv') # Se le agregó pre-processed
    df = pd.read_csv(path) 
    df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
    # change 7 to 2
    df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
    
    df.to_csv(os.path.join(data_config['root'], 'no_sample_df.csv'))

    # oversample fear
    fear_df = df[df['label']==3]
    for i in range(30):
        df = df.append(fear_df)

    sur_df = df[df['label']==4]
    for i in range(10):
        df = df.append(sur_df)
        
    df.to_csv(os.path.join(data_config['root'], 'modified_df.csv'))

    #emotion_dict = {'ang': 0,
    #            'hap': 1,
    #            'sad': 2,
    #            'neu': 3,}

    scalar = MinMaxScaler() # Todo esto es para estandarizar/escalar los features
    df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])
    print("\nNew labels on audio feature dataframe")
    print(df.head())

    audio_train, audio_test = train_test_split(df, test_size=0.20)
    audio_train.to_csv(os.path.join(data_config['audio'],'audio_train.csv'), index=False)
    audio_test.to_csv(os.path.join(data_config['audio'],'audio_test.csv'), index=False)

    print("\nFinished preparation of audio data!")
    print(audio_train.shape, audio_test.shape)
    return audio_train, audio_test


# Clean transcriptions
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def transcribe_sessions():
    file2transcriptions = {}
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    for sess in range(1, 6):
        transcript_path = os.path.join(data_config['IEMO'], 'Session{}/dialog/transcriptions/'.format(sess))
        for f in os.listdir(transcript_path):
            with open('{}{}'.format(transcript_path, f), 'r', encoding="latin-1") as f: # Agregamos encoding
                all_lines = f.readlines()
            for l in all_lines:
                audio_code = useful_regex.match(l) # Original pero tenia none en algunos
                if audio_code:
                    audio_code = useful_regex.match(l).group()
                    transcription = l.split(':')[-1].strip()
                    # assuming that all the keys would be unique and hence no `try`
                    file2transcriptions[audio_code] = transcription
    
    path = os.path.join(data_config["text"], 'audiocode2text.pkl')
    with open(path, 'wb') as file:
        pickle.dump(file2transcriptions, file)
    return file2transcriptions


def prepare_text_data(audiocode2text, x_train, x_test):
    # Prepare text data
    text_train = pd.DataFrame()
    text_train['wav_file'] = x_train['wav_file']
    text_train['label'] = x_train['label']
    text_train['transcription'] = [normalizeString(audiocode2text[code])for code in x_train['wav_file']]

    text_test = pd.DataFrame()
    text_test['wav_file'] = x_test['wav_file']
    text_test['label'] = x_test['label']
    text_test['transcription'] = [normalizeString(audiocode2text[code]) for code in x_test['wav_file']]

    text_train.to_csv(os.path.join(data_config['text'],'trans_text_train.csv'), index=False)
    text_test.to_csv(os.path.join(data_config['text'],'trans_text_test.csv'), index=False)

    print("\nFinished transcription of text files!")
    print(text_train.shape, text_test.shape)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    train_features = tfidf.fit_transform(text_train.transcription).toarray()
    test_features = tfidf.fit_transform(text_test.transcription).toarray()

    text_train['transcription'] = train_features
    text_test['transcription'] = test_features

    text_train.to_csv(os.path.join(data_config['text'],'text_train.csv'), index=False)
    text_test.to_csv(os.path.join(data_config['text'],'test_train.csv'), index=False)

    print("\nFinished preparation of text files!")
    print(text_train.shape, text_test.shape)

def prepare_combined_data():
    # Agarramos a partir de la col 2 para evitar label y wavfile.
    x_train_text = pd.read_csv(os.path.join(data_config['text'],'trans_text_train.csv'))
    x_test_text = pd.read_csv(os.path.join(data_config['text'],'trans_text_test.csv'))

    x_train_audio = pd.read_csv(os.path.join(data_config['audio'],'audio_train.csv'))
    x_test_audio = pd.read_csv(os.path.join(data_config['audio'],'audio_test.csv'))

    y_train_audio = x_train_audio['label']
    y_test_audio = x_test_audio['label']

    y_train = y_train_audio  # since y_train_audio == y_train_text
    y_test = y_test_audio  # since y_train_audio == y_train_text

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    # Pasamos las transcripciones a features.
    features_text = tfidf.fit_transform(x_train_text.append(x_test_text).transcription).toarray()
    
    x_train_text = features_text[:x_train_text.shape[0]]
    x_test_text = features_text[-x_test_text.shape[0]:]

    # Guardamos a partir de columna 2 porque esa no tiene el label.
    combined_x_train = np.concatenate((np.array(x_train_audio[x_train_audio.columns[2:]]), x_train_text), axis=1)
    combined_x_test = np.concatenate((np.array(x_test_audio[x_test_audio.columns[2:]]), x_test_text), axis=1)

    x_train_df = pd.DataFrame(columns=['label','features'])
    x_train_df['label'] = y_train
    x_train_df['features'] = combined_x_train
    x_train_df.to_csv(os.path.join(data_config["combined"],'combined_train.csv'), index=False)

    x_test_df = pd.DataFrame(columns=['label','features'])
    x_test_df['label'] = y_test
    x_test_df['features'] = combined_x_test
    x_test_df.to_csv(os.path.join(data_config["combined"],'combined_test.csv'), index=False)

    combined_features_dict = {}

    combined_features_dict['x_train'] = combined_x_train
    combined_features_dict['x_test'] = combined_x_test
    combined_features_dict['y_train'] = np.array(y_train)
    combined_features_dict['y_test'] = np.array(y_test)

    with open(os.path.join(data_config['combined'], 'combined_features.pkl'), 'wb') as f:
        pickle.dump(combined_features_dict, f)

    print("\nFinished preparation of combined features!")
    print(combined_x_train.shape, combined_x_test.shape)

def main():
    audio_train, audio_test = prepare_audio_files()
    processed_text = transcribe_sessions()
    prepare_text_data(processed_text, audio_train, audio_test)
    prepare_combined_data()

if __name__ == '__main__':
    main()