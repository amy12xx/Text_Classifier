#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import codecs
import os
import nltk
from nltk import FreqDist
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import bigrams
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.classify.naivebayes import NaiveBayesClassifier
import collections


CB_STOP_WORDS = nltk.corpus.stopwords.words('english_extended_for_cb')
ENG_STOP_WORDS = nltk.corpus.stopwords.words('english')

def tokenize(text):
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)
	return tokens

def pos_tag(tokens):
	pos_tags = nltk.pos_tag(tokens)
	return pos_tags

def lemmatize(pos_tags):
	lem = WordNetLemmatizer()
	lemmas = [lem.lemmatize(tk, g[0].lower()) if g[0].lower() in ['a','n','v'] else lem.lemmatize(tk) for (tk, g) in pos_tags]
	return lemmas

def remove_common_words(tokens):
	clean_token_list = []
	for token in tokens:
		if token.lower() not in nltk.corpus.stopwords.words('english_extended_for_cb'):
			clean_token_list.append(token.lower())
	return clean_token_list

def lower(tokens):
	tokens = [t.lower() for t in tokens]
	return tokens

def bigrammeasure(tokens):
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	finder = BigramCollocationFinder.from_words(lower(tokens))
	finder.apply_freq_filter(2)
	collocation = finder.nbest(bigram_measures.likelihood_ratio, 10)
	collocation2 = collocation[:]
	for c in collocation:
		if c[0] in ENG_STOP_WORDS or c[1] in ENG_STOP_WORDS:
			collocation2.remove(c)
	return collocation2

def clean_up_text(text):
	#Strip of tags from text using Beautiful Soup
	soup = BeautifulSoup(text)
	soup = soup.get_text()
	soup = soup.strip('\n').strip("\t").strip("\r").replace("\n"," ").replace("\t"," ").replace("\r"," ")
	return soup


def bag_of_words(words):
	bw = {}
   	for word in words:
   		word = str(word).encode('utf-8')
   		bw[word] = True
   	return bw
