#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import codecs
import os
import sys
import random
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors, datasets
import numpy as np
from time import time
from sklearn.utils.extmath import density
from sklearn.grid_search import GridSearchCV
from text_processor import clean_up_text
import cPickle as pickle

all_course_data = []

def process_file_to_predict(filename):
	data = []
	with codecs.open(filename, 'r') as course_file:
		for line in course_file:
			words = {}
			course = line.split('^')
			title1 = course[0] if course[0] else ''
			title2 = course[1] if course[1] else ''
			description = course[2] if course[2] else ''
			course_text_str = title1 + " " + title2 + " " + description
			soup = clean_up_text(course_text_str)
			data.append(soup)
	return data

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    #svm = SVC(C=1.0, kernel='linear')
    svm = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='crammer_singer', \
    	fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
    t0 = time()
    svm.fit(X, y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    return svm

def load_data_sets():
	"""
    Load training data set
    """
	all_course_data = pickle.load(open('Documents/pythonfiles/CB_scripts/Text_Classifier/data/course_corpus.pkl', 'rb'))
	return all_course_data

def SVMClassify():
	all_course_data = load_data_sets()
	y = [d[0] for d in all_course_data]
	corpus = [d[1] for d in all_course_data]

	vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
	X = vectorizer.fit_transform(corpus)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
	svm = train_svm(X_train, y_train)
	pred = svm.predict(X_test)
	#svm = train_svm(X,y)

	print 'confusion matrix'
	print(svm.score(X_test, y_test))
	print(confusion_matrix(pred, y_test))
	print cross_validation.cross_val_score(svm, X, y, scoring='accuracy')
 	score = metrics.f1_score(y_test, pred)
 	print("f1-score:   %0.3f" % score)

 	########### Test unclassified data ########################
	# unclassified_file = 'Documents/pythonfiles/CB_scripts/Text_Classifier/data/unclassified_courses.txt'
	# corpus2 = [d for d in process_file_to_predict(unclassified_file)]
	# X2 = vectorizer.transform(corpus2)
	# t0 = time()
	# pred2 = svm.predict(X2)
	# test_time = time() - t0
	# print("test time:  %0.3fs" % test_time)

	########### save classified data ##########################
	# classified_file = 'Documents/pythonfiles/CB_scripts/Text_Classifier/data/classified_courses.txt'
	# with codecs.open(classified_file, 'w', encoding='utf-8') as final_file:
	# 	for p in pred2:
	# 		final_file.write(p + '\n')

def MultionomialNBClassify():
	clf = MultinomialNB(alpha=0.8, class_prior=None, fit_prior=True)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	print 'confusion matrix'
	print(clf.score(X_test, y_test))
	print(confusion_matrix(pred, y_test))

def grid_search_SVC():
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

	all_course_data = load_data_sets()
	y = [d[0] for d in all_course_data]
	corpus = [d[1] for d in all_course_data]

	vectorizer = TfidfVectorizer(min_df=1)
	X = vectorizer.fit_transform(corpus)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

	svm = SVC()
	clf = GridSearchCV(svm, parameters)
	clf.fit(X_train, y_train)
	print 'Grid search SVC'
	print clf.best_score_
	best_parameters = clf.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print "\t%s: %r" % (param_name, best_parameters[param_name])

def grid_search_LinearSVC():
	parameters = {'C':[1, 10], 'multi_class':('crammer_singer', 'ovr'), 'class_weight':(None, 'auto'), 'dual':(True, False)}

	all_course_data = load_data_sets()
	y = [d[0] for d in all_course_data]
	corpus = [d[1] for d in all_course_data]

	vectorizer = TfidfVectorizer(min_df=1)
	X = vectorizer.fit_transform(corpus)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

	svm = LinearSVC(tol=0.0001, penalty = 'l2', loss = 'l2', fit_intercept=True, intercept_scaling=1, verbose=0, random_state=None)
	clf = GridSearchCV(svm, parameters)
	clf.fit(X_train, y_train)
	print 'Grid search Linear SVC'
	print clf.best_score_
	best_parameters = clf.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print "\t%s: %r" % (param_name, best_parameters[param_name])

def kNN():
	all_course_data = load_data_sets()
	y = [d[0] for d in all_course_data]
	corpus = [d[1] for d in all_course_data]

	vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
	X = vectorizer.fit_transform(corpus)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
	clf = neighbors.KNeighborsClassifier(2, weights='distance')
	clf.fit(X_train,y_train)
	pred = clf.predict(X_test)
	print 'confusion matrix nearest neighbours'
	print(clf.score(X_test, y_test))
	print(confusion_matrix(pred, y_test))


def main():
	#SVMClassify()
	kNN()
	#MultinomialNBClassify()
	#grid_search_SVC()
	#grid_search_LinearSVC()

if __name__ == '__main__':
	main()