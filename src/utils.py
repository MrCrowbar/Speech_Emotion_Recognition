import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from config import data_config 
import os
import pickle

def plot_confusion_matrix(model, cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(model)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    file_name = model + '_' +  str(data_config['mode']) + '.png'
    plt.savefig(file_name)
    plt.show()
    
def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels

def display_results(model,y_test, pred_probs, cm=True):
    emo_keys = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])
    pred = np.argmax(pred_probs, axis=-1)
    print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
    print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
    print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
    if cm:
        plot_confusion_matrix(model, confusion_matrix(y_test, pred), classes=emo_keys)

def show_all(pred_path, y_test, cmap=plt.cm.Blues):
    classes = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])
    predictions = []
    names = []
    axes = [[0, 0],[0, 1],[0, 2],[1, 0],[1, 1],[1, 2], [0,3]]
    
    for pred in os.listdir(pred_path):
        with open(os.path.join(pred_path,pred), 'rb') as f:
                file = pickle.load(f)
                names.append(pred)
                predictions.append(file)

    tuples = zip(axes, predictions)
    tick_marks = np.arange(len(classes))
    get_matrixes(tuples, classes, names, tick_marks, y_test)


def get_matrixes(tuples, classes, names, tick_marks, y_test, cmap=plt.cm.Blues):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))
    index = 0
    for ax, p in tuples:
        a = ax[0]
        b = ax[1]
        plt.sca(axs[a,b])
        pred = np.argmax(p, axis=-1)
        cm = confusion_matrix(y_test,pred)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.title(names[index])
        plt.imshow(confusion_matrix(y_test,pred), interpolation='nearest', cmap=cmap)
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        index += 1
    fig.tight_layout()
    file_name = 'mosaico_' +  str(data_config['mode']) + '.png'
    plt.savefig(file_name)
    plt.show()
