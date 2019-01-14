import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(cm, classes, xname, yname,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(yname)
    plt.xlabel(xname)


parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-g', help="ground_truth folder path with pulled json from auto_train")
parser.add_argument('-m', help="model.out file with probabilities")

args = parser.parse_args()

g = args.g.rstrip('/')
m = args.m.rstrip('/')
gnd = pd.read_csv(g)
model = pd.read_csv(m)

gnd = pd.DataFrame(gnd, columns=["name", "gnd_label"])
model = pd.DataFrame(model, columns=["name", "pred_label"])

final = pd.merge(left=gnd, right=model, on='name', how='right', suffixes=('_annos', '_preds'),
                 sort=True)
final = final.dropna()
y_test = final['gnd_label'].values
y_pred = final['pred_label'].values
labels = np.unique(list(np.unique(y_test)) + list(np.unique(y_pred)))
cnf = confusion_matrix(y_test, y_pred, labels)

print classification_report(y_test, y_pred, labels=labels)
plot_confusion_matrix(cnf, classes=labels, yname='ground_truth',
                      xname='model_predictions',
                      title="confusion matrix")
plt.savefig('confusion_matrix_23_07_2018.png')
plt.show()
