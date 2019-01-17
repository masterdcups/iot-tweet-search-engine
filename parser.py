import csv
import numpy as np
import re

import nltk
from spellchecker import SpellChecker
import preprocessor
import gensim


def replace_abbreviations(tokens):
	j = 0
	file_name = "corpus/slang.txt"
	with open(file_name, 'r') as myCSVfile:
		# Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
		data_from_file = csv.reader(myCSVfile, delimiter="=")
		for token in tokens:
			# Removing Special Characters.
			_token = re.sub('[^a-zA-Z0-9-_.]', '', token)
			for row in data_from_file:
				# Check if selected word matches short forms[LHS] in text file.
				if token.upper() == row[0]:
					# If match found replace it with its Abbreviation in text file.
					tokens[j] = row[1]
			j = j + 1
		myCSVfile.close()
	return gensim.utils.simple_preprocess(' '.join(tokens))


def remove_stopwords_spelling_mistakes(spell, tokens):
	clean_tokens = []
	for token in tokens:
		# correction of spelling mistakes
		token = spell.correction(token)
		if token not in nltk.corpus.stopwords.words('english'):
			clean_tokens.append(token)
	return clean_tokens


class Parser:

	@staticmethod
	def clean_tweet(tweet_text):
		"""
		Taking a raw tweet, return a cleaned list of tweets tokens
		:param tweet_text:
		:return: array of tokens words
		"""

		# todo : find another solution for nltk download !
		import ssl

		try:
			_create_unverified_https_context = ssl._create_unverified_context
		except AttributeError:
			pass
		else:
			ssl._create_default_https_context = _create_unverified_https_context

		nltk.download('stopwords')

		# load spell checker
		spell = SpellChecker()

		# load lemmatizer
		# lmtzr = WordNetLemmatizer()

		tokens = []
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.MENTION, preprocessor.OPT.RESERVED,
								 preprocessor.OPT.EMOJI, preprocessor.OPT.SMILEY)
		tweet = preprocessor.clean(tweet_text)
		hashtags = list(part[1:] for part in tweet.split() if part.startswith('#'))
		tokens += gensim.utils.simple_preprocess(tweet) + gensim.utils.simple_preprocess(' '.join(hashtags))

		tokens = replace_abbreviations(tokens)
		tokens = remove_stopwords_spelling_mistakes(spell, tokens)
		# lemmatized_tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

		return tokens

	@staticmethod
	def parsing_iot_corpus(path):
		# Les lignes en commentaires devront être décommentées lorsque le corpus sera complété avec le texte des tweets et les vecteurs associés
		# Les deux lignes avant "fichier.close()" devront alors être supprimées
		map = {}
		with open(path, "r") as fichier:
			line = fichier.readline().replace('\n', '').split('\t')
			for key in line:
				map[key] = []
			map['Vector'] = []
			for line in fichier:
				tweet = line.replace('\n', '').split("\t")
				map['TweetID'] += [tweet[0]]
				map['Sentiment'] += [tweet[1]]
				map['TopicID'] += [tweet[2]]
				map['Country'] += [tweet[3]]
				map['Gender'] += [tweet[4]]
				# map['URLs'] += tweet[5:-2]
				# map['Text'] += Parser.clean_tweet(tweet[-2])
				# map['Vector'] += tweet[-1]
				map['URLs'] += [tweet[5:]]
				map['Vector'] += [np.zeros(300)]
		fichier.close()
		return map
