import argparse
import itertools
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc


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
classes = ['normal', 'abnormal']

args = parser.parse_args()
g = args.g.rstrip('/')
m = args.m.rstrip('/')
gnd = pd.read_csv(g)
model = pd.read_csv(m)

final = pd.merge(left=gnd, right=model, on='name', how='right', suffixes=('_annos', '_preds'),
                 sort=True)
final = final.dropna()
y_test = np.zeros((len(final), 2))
y_score = np.zeros((len(final), 2))
print final
for idx, i in enumerate(classes):
    y_test[:, idx] = final[i + '_annos'].values
    y_score[:, idx] = final[i + '_preds'].values
n_classes = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in xrange(2):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

lw = 2
# Plot all ROC curves
plt.figure()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for the OCT scans classification')
plt.legend(loc="lower right")
plt.savefig('OCT_roc.jpg')
plt.show()
