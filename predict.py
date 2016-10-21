#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import sys
import prepare

VALUES = joblib.load('values.pkl')
model = joblib.load('model.pkl')

prepare.get_images('test_vaporwave')

vaporwave_result = model.predict(prepare.get_images('test_vaporwave'))
garbage_result = model.predict(prepare.get_images('test_garbage'))

print 'Results for test『ＶＡＰＯＲＷＡＶＥ』', map(VALUES.__getitem__, vaporwave_result)
print 'Results for 「ＧＡＲＢＡＧＥ」', map(VALUES.__getitem__, garbage_result)
