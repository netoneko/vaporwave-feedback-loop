#!/usr/bin/env python

import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from IPython import embed
from skimage import io
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray

import sys, os, glob

VALUES = {
    'VAPORWAVE': 0,
    'GARBAGE': 1
}

def prepare_image(filename):
    image = resize(rgb2gray(io.imread(filename)), [640, 480])
    return np.array(hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False,
        normalise=None))

def get_images(key):
    return map(prepare_image, glob.glob(os.path.join('./images/%s/*' % key)))

def get_features(key):
    images = get_images(key)
    features = images

    print features[0]
    # TODO: add more features

    answers = [VALUES[key.upper()]] * len(features)

    return features, answers


def build_model(features, answers):
    model = LogisticRegression()
    model.fit(features, answers)

    return model


def test_type(key):
    test_features, test_answers = get_features(key)

    # test_features = vectorizer.transform(test_lines)
    test_predictions = model.predict(test_features)

    return accuracy_score(test_answers, test_predictions)


def test_model(model):
    print 'accuracy on same set', accuracy_score(answers, model.predict(features))
    return map(test_type, VALUES.keys())

def merge_features(data):
    merged_features = []
    merged_answers = []

    print data[0]

    for features, answers in data:
        print features, answers

        merged_features = merged_features + features
        merged_answers = merged_answers + answers

    return np.array(merged_features), np.array(merged_answers)

if __name__ == '__main__':
    print 'Building models...'

    features, answers = merge_features([get_features('vaporwave'), get_features('garbage')])

    while(True):
        model = build_model(features, answers)
        results = test_model(model)

        joblib.dump(model, 'model.pkl')
        joblib.dump({v: k for k, v in VALUES.items()}, 'values.pkl')

        sys.exit(0)
