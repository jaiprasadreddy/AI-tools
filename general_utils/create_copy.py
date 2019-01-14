import glob
import os
import sys

import cv2

copies = {"NonGradable": 4, "Normal": 1, "NPDR": 1, "Others": 4, "PDR": 2}

# path = sys.argv[1]
path = sys.argv[1]
output = sys.argv[2]
for label in copies:
    iter = copies[label]
    path1 = path.rstrip('/') + '/' + label + '/*.jpg'
    imgs = glob.glob(path1)
    print imgs
    out_path = output.rstrip('/') + '/' + label
    os.makedirs(out_path)
    for img in imgs:
        name = img.split('/')[-1]
        img = cv2.imread(img)
        for i in xrange(iter):
            cv2.imwrite(out_path + '/' + name.split('.')[0] + '_' + str(i) + '.jpg', img)
