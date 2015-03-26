#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle
import os
import codecs
from text_processor import clean_up_text

all_course_data = []

def process_penn_wsu_file(filename):
	data = []
	with codecs.open(filename, 'r', encoding='utf-8') as course_file:
		for line in course_file:
			words = []
			course = line.split('^')
			category = course[0] if course[0] else ''
			description = course[1] if course[1] else ''
			course_text_str = description
			soup = clean_up_text(course_text_str)
   			data = category, soup
			all_course_data.append(data)
	return all_course_data

def process_file(filename):
	data = []
	with codecs.open(filename, 'r', encoding='utf-8') as course_file:
		for line in course_file:
			words = {}
			course = line.split('^')
			category = course[0] if course[0] else ''
			cb_title = course[1] if course[1] else ''
			provider_title = course[2] if course[2] else ''
			description = course[3] if course[3] else ''
			course_text_str = cb_title + " " + provider_title + " " + description
			soup = clean_up_text(course_text_str)
   			data = category, soup
			all_course_data.append(data)
	return all_course_data

def load_data_sets():
	"""
    Load training data sets
    """
	basefilename = 'Documents/pythonfiles/CB_scripts/Text_Classifier/data/'
	training_data_set = ['wsu_courses.txt', 'penn_courses.txt', 'uw_courses.txt', 'berkeley_courses.txt']

	all_course_data = process_file(basefilename+"course_corpus2.txt")
	for data_set in training_data_set:
		filename = str(basefilename + data_set)
		all_course_data = process_penn_wsu_file(filename)

def main():
	load_data_sets()
	output = open('Documents/pythonfiles/CB_scripts/Text_Classifier/data/course_corpus.pkl', 'wb')
	pickle.dump(all_course_data, output)

if __name__ == '__main__':
	main()