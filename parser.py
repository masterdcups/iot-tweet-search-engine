import csv
import re

import gensim
import nltk
import numpy as np
import preprocessor
from spellchecker import SpellChecker


def replace_abbreviations(tokens):
	"""
	Replace the abbreviations (OMG -> Oh My God) based on the dictionary in slang.txt
	:param tokens: words of the tweet
	:return: words with abbreviations replaced by their meaning
	"""
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
	"""
	Remove stopwords and corrects spelling mistakes
	:param spell: Object to correct spelling mistakes
	:param tokens: words of the tweet
	:return: words cleaned and corrected
	"""
	clean_tokens = []
	for token in tokens:
		# correction of spelling mistakes
		token = spell.correction(token)
		if token not in nltk.corpus.stopwords.words('english'):
			clean_tokens.append(token)
	return clean_tokens


def tweet2vec(tweet_text, model):
	sentence_vector = []

	for word in tweet_text:
		try:
			sentence_vector.append(model.wv[word])

		except KeyError:
			pass

	# if a tweet word do not appear in the model we put a zeros vector
	if len(sentence_vector) == 0:
		sentence_vector.append(np.zeros_like(model.wv["tax"]))

	return np.mean(sentence_vector, axis=0)


class Parser:

	def __init__(self):
		self.load_nltk()

	def clean_tweet(self, tweet_text):
		"""
		Taking a raw tweet, return a cleaned list of tweets tokens
		:param tweet_text:
		:return: array of tokens words
		"""

		# load spell checker
		spell = SpellChecker()

		tokens = []
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.MENTION, preprocessor.OPT.RESERVED,
								 preprocessor.OPT.EMOJI, preprocessor.OPT.SMILEY)
		tweet = preprocessor.clean(tweet_text)
		hashtags = list(part[1:] for part in tweet.split() if part.startswith('#'))
		tokens += gensim.utils.simple_preprocess(tweet) + gensim.utils.simple_preprocess(' '.join(hashtags))

		tokens = replace_abbreviations(tokens)
		tokens = remove_stopwords_spelling_mistakes(spell, tokens)

		return tokens

	@staticmethod
	def parsing_iot_corpus(corpus_path):
		"""
		Parse the corpus and return the list of tweets with characteristics
		:param corpus_path: path of the corpus
		:return: array of dict (tweets)
		"""
		parser = Parser()

		tweets = []

		with open(corpus_path, "r") as file:
			file.readline()

			for line in file:
				tweet = line.replace('\n', '').split("\t")
				tweet_infos = {}
				tweet_infos['TweetID'] = tweet[0]
				tweet_infos['Sentiment'] = tweet[1]
				tweet_infos['TopicID'] = tweet[2]
				tweet_infos['Country'] = tweet[3]
				tweet_infos['Gender'] = tweet[4]
				tweet_infos['URLs'] = tweet[5:-3]
				tweet_infos['Text'] = parser.clean_tweet(tweet[-3])
				tweet_infos['Author'] = tweet[-2]
				tweet_infos['Vector'] = np.asarray([float(x) for x in tweet[-1][1:-1].split(', ')])
				tweets.append(tweet_infos)

		file.close()
		return tweets

	def get_composant(self, column):
		pass

	@staticmethod
	def add_vector_to_corpus(corpus_path, new_corpus_path, write_every=1000):
		"""
		Create a new Vector column on the corpus
		:param write_every: write in the final file every x lines
		:param corpus_path:
		:param new_corpus_path:
		:return:
		"""
		parser = Parser()
		model = gensim.models.KeyedVectors.load_word2vec_format('corpus/GoogleNews-vectors-negative300.bin',
																binary=True)

		print('GoogleNews-vectors LOADED')

		corpus = open(corpus_path, 'r', encoding='utf-8')
		new_corpus = open(new_corpus_path, 'w', encoding='utf-8')

		lines = corpus.readlines()
		corpus.close()
		new_lines = []
		last_written = -1
		new_lines.append(lines[0][:-1] + '\tVector\n')

		for i in range(1, len(lines)):

			new_lines.append(lines[i][:-1] + '\t' + str(
				list(tweet2vec(parser.clean_tweet(lines[i].split('\t')[-2]), model))) + '\n')

			if i % write_every == 0:
				new_corpus.write(''.join(new_lines[(last_written + 1):]))
				last_written = i
				print(str(last_written) + '/' + str(len(lines)) + ' treated')

		new_corpus.write(''.join(new_lines[(last_written + 1):]))
		new_corpus.close()

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


if __name__ == '__main__':
	# Parser.add_vector_to_corpus('corpus/fake-iot-corpus2.tsv', 'corpus/test.tsv', write_every=3)
	Parser.add_vector_to_corpus('corpus/iot-tweets-2009-2016-complet.tsv', 'corpus/iot-tweets-vector.tsv')
