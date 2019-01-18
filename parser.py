import csv
import re

import gensim
import nltk
import numpy as np
import preprocessor
from spellchecker import SpellChecker


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

	def __init__(self):
		self.load_nltk()

	@staticmethod
	def clean_tweet(tweet_text):
		"""
		Taking a raw tweet, return a cleaned list of tweets tokens
		:param tweet_text:
		:return: array of tokens words
		"""

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
		parser = Parser()

		tweets = []

		with open(path, "r") as file:
			file.readline()

			for line in file:
				tweet = line.replace('\n', '').split("\t")
				tweet_infos = {}
				tweet_infos['TweetID'] = tweet[0]
				tweet_infos['Sentiment'] = tweet[1]
				tweet_infos['TopicID'] = tweet[2]
				tweet_infos['Country'] = tweet[3]
				tweet_infos['Gender'] = tweet[4]
				tweet_infos['URLs'] = tweet[5:-2]
				tweet_infos['Text'] = parser.clean_tweet(tweet[-2])
				tweet_infos['Vector'] = np.asarray([float(x) for x in tweet[-1][1:-1].split(', ')])
				tweets.append(tweet_infos)

		file.close()
		return tweets

	def get_composant(self, column):
		pass

	def load_nltk(self):
		# todo : find another solution for nltk download !
		import ssl

		try:
			_create_unverified_https_context = ssl._create_unverified_context
		except AttributeError:
			pass
		else:
			ssl._create_default_https_context = _create_unverified_https_context

		nltk.download('stopwords')
