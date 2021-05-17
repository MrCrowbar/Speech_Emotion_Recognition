import pandas as pd
import xgboost as xgb
import pickle
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from config import ensemble_config, data_config
from utils import display_results, show_all

emotion_dict = {'ang': 0,
                'hap': 1,
                'sad': 2,
                'fea': 3,
                'sur': 4,
                'neu': 5}

emo_keys = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])

mode_dict = {0:'audio', 1:'text', 2:'combined'}
mode = mode_dict[data_config["mode"]] # Default mode is 0:audio
data_path = os.path.join('../data', mode, mode) # Cambiar esto de mode mode

x_train = ''
x_test = ''
y_train = ''
y_test = ''

if mode == "combined":
    x_train = pd.read_csv(data_path + '_train.csv')
    x_test = pd.read_csv(data_path + '_test.csv')
    y_train = x_train['label']
    y_test = x_test['label']
    del x_train['label']
    del x_test['label']
else:
    x_train = pd.read_csv(data_path + '_train.csv')
    x_test = pd.read_csv(data_path + '_test.csv')
    y_train = x_train['label']
    y_test = x_test['label']
    del x_train['label']
    del x_test['label']
    del x_train['wav_file']
    del x_test['wav_file']


def random_forest(matrixes,pred_path, weights_path, type):
    # Train model
    rf_classifier = RandomForestClassifier(n_estimators=1200, min_samples_split=25)
    rf_classifier.fit(x_train, y_train)

    # Save model
    model_file = os.path.join(weights_path,type,"rf_classifier.pkl")
    pickle.dump(rf_classifier,open(model_file, 'wb'))

    # Predict
    pred_probs = rf_classifier.predict_proba(x_test)
    matrixes.append(confusion_matrix(y_test,np.argmax(pred_probs, axis=-1)))
    
    # Results
    model_name = "Random Forest"
    display_results(model_name,y_test, pred_probs)
    
    pred_file = os.path.join(pred_path,type,"rf_classifier.pkl")
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_probs, f)

def xgb_classifier(matrixes,pred_path, weights_path, type):
    # Train model
    xgb_classifier = xgb.XGBClassifier(max_depth=7, learning_rate=0.008, objective='multi:softprob', 
                                   n_estimators=1200, sub_sample=0.8, num_class=len(emotion_dict),
                                   booster='gbtree', n_jobs=4)
    xgb_classifier.fit(x_train, y_train)
    
    # Save model
    model_file = os.path.join(weights_path,type,"xgb_classifier.pkl")
    pickle.dump(xgb_classifier,open(model_file, 'wb'))

    # Predict
    pred_probs = xgb_classifier.predict_proba(x_test)
    matrixes.append(confusion_matrix(y_test,np.argmax(pred_probs, axis=-1)))

    # Results
    model_name = "XGB classifier"
    display_results(model_name, y_test, pred_probs)

    pred_file = os.path.join(pred_path,type,"xgb_classifier.pkl")
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_probs, f)

def mlp_classifier(matrixes,pred_path, weights_path, type):
    # Train model
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(650, ), activation='relu', solver='adam', alpha=0.0001,
                                batch_size='auto', learning_rate='adaptive', learning_rate_init=0.01,
                                power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
                                verbose=False, warm_start=True, momentum=0.8, nesterovs_momentum=True,
                                early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-08)

    mlp_classifier.fit(x_train, y_train)

    # Save model
    model_file = os.path.join(weights_path,type,"mlp_classifier.pkl")
    pickle.dump(mlp_classifier,open(model_file, 'wb'))

    # Predict
    pred_probs = mlp_classifier.predict_proba(x_test)
    matrixes.append(confusion_matrix(y_test,np.argmax(pred_probs, axis=-1)))

    # Results
    model_name = "MLP classifier"
    display_results(model_name, y_test, pred_probs)

    pred_file = os.path.join(pred_path,type,"mlp_classifier.pkl")
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_probs, f)

def mnb_classifier(matrixes,pred_path, weights_path, type):
    # Train model
    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(x_train, y_train)

    # Save model
    model_file = os.path.join(weights_path,type,"mnb_classifier.pkl")
    pickle.dump(mnb_classifier,open(model_file, 'wb'))

    # Predict
    pred_probs = mnb_classifier.predict_proba(x_test)
    matrixes.append(confusion_matrix(y_test,np.argmax(pred_probs, axis=-1)))

    # Results
    model_name = "MNB classifier"
    display_results(model_name, y_test, pred_probs)

    pred_file = os.path.join(pred_path,type,"mnb_classifier.pkl")
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_probs, f)

def lr_classifier(matrixes,pred_path, weights_path, type):
    # Train model
    lr_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    lr_classifier.fit(x_train, y_train)

    # Save model
    model_file = os.path.join(weights_path,type,"lr_classifier.pkl")
    pickle.dump(lr_classifier,open(model_file, 'wb'))

    # Predict
    pred_probs = lr_classifier.predict_proba(x_test)
    matrixes.append(confusion_matrix(y_test,np.argmax(pred_probs, axis=-1)))

    # Results
    model_name = "LR classifier"
    display_results(model_name, y_test, pred_probs)

    pred_file = os.path.join(pred_path,type,"lr_classifier.pkl")
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_probs, f)

def ensemble_classifier(matrixes, pred_path, type):
    predictions = []
    models = []
    pred_file = os.path.join(pred_path,type,'')
    for model, flag in ensemble_config.items():
        if flag:
            with open(pred_file + '{}_classifier.pkl'.format(model), 'rb') as f:
                file = pickle.load(f)
                predictions.append(file)
                models.append(model)

    # Predict
    pred_probs = sum(predictions) / len(predictions)
    matrixes.append(confusion_matrix(y_test,np.argmax(pred_probs, axis=-1)))
    combination = '_'.join(models) + "_classifier.pkl"
    pred_file = os.path.join(pred_path,type,combination)
    with open(pred_file, 'wb') as f:
        pickle.dump(pred_probs, f)
    display_results("Ensemble", y_test, pred_probs)

def trainAll(matrixes,pred_path, weights_path, type):
    random_forest(matrixes,pred_path, weights_path, type)
    xgb_classifier(matrixes,pred_path, weights_path, type)
    mlp_classifier(matrixes,pred_path, weights_path, type)
    mnb_classifier(matrixes,pred_path, weights_path, type)
    lr_classifier(matrixes,pred_path, weights_path, type)
    ensemble_classifier(matrixes, pred_path, type)

if __name__ == '__main__':
    weights_path = data_config["trained_models"]
    pred_path = data_config["saved_preds"]
    type = mode
    matrixes = []
    
    if data_config['train']:
        trainAll(matrixes, pred_path, weights_path, type)

    if data_config['test']:
        show_all(os.path.join(pred_path, type, ''), y_test)