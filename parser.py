import os

import gensim
import nltk
import numpy as np
import pandas as pd
import preprocessor
import scipy.sparse as sp
from spellchecker import SpellChecker

from definitions import ROOT_DIR


class Parser:

	def __init__(self):
		self.load_nltk()
		self.model = None
		self.abbreviations = None
		self.spell_check = None

		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.MENTION, preprocessor.OPT.RESERVED,
								 preprocessor.OPT.EMOJI, preprocessor.OPT.SMILEY)

	def clean_tweet(self, tweet_text):
		"""
		Taking a raw tweet, return a cleaned list of tweets tokens
		:param tweet_text:
		:return: array of tokens words
		"""

		tweet = preprocessor.clean(tweet_text)
		tokens = [word[1:] if word.startswith('#') else word for word in tweet.split(' ')]

		tokens = self.replace_abbreviations(tokens)
		tokens = self.remove_stopwords_spelling_mistakes(tokens)
		tokens = gensim.utils.simple_preprocess(' '.join(tokens))

		return tokens

	def load_spell_check(self):
		if self.spell_check is not None:
			return

		self.spell_check = SpellChecker()

	def load_abbreviations(self):
		if self.abbreviations is not None:
			return

		file_name = "corpus/slang.txt"
		file = open(file_name, 'r')
		self.abbreviations = {}
		for line in file.readlines():
			parts = line[:-1].split('=')
			self.abbreviations[parts[0].upper()] = parts[1]

		file.close()

	def replace_abbreviations(self, tokens):
		"""
		Replace the abbreviations (OMG -> Oh My God) based on the dictionary in slang.txt
		:param tokens: words of the tweet
		:return: words with abbreviations replaced by their meaning
		"""

		self.load_abbreviations()

		for i in range(len(tokens)):
			tokens[i] = self.abbreviations[tokens[i]] if tokens[i] in self.abbreviations else tokens[i]

		return tokens

	def remove_stopwords_spelling_mistakes(self, tokens):
		"""
		Remove stopwords and corrects spelling mistakes
		:param spell: Object to correct spelling mistakes
		:param tokens: words of the tweet
		:return: words cleaned and corrected
		"""

		# self.load_spell_check()

		return list(filter(lambda token: token not in nltk.corpus.stopwords.words('english'), tokens))

	@staticmethod
	def parsing_iot_corpus(corpus_path):
		"""
		Parse the corpus and return the list of tweets with characteristics
		:param corpus_path: path of the corpus
		:return: array of dict (tweets)
		"""

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
				tweet_infos['URLs'] = tweet[5:-4]
				tweet_infos['Text'] = tweet[-4]
				tweet_infos['Author'] = tweet[-3]
				tweet_infos['CleanedText'] = tweet[-2]
				tweet_infos['Vector'] = np.asarray([float(x) for x in tweet[-1][1:-1].split(', ')])
				tweets.append(tweet_infos)

		file.close()
		return tweets

	@staticmethod
	def parsing_iot_corpus_pandas(corpus_path, separator='\t', categorize=False):
		"""
		Parse the corpus and return a Pandas DataFrame
		:param categorize: boolean to make the tweet and user ids start to 0
		:param separator:
		:param corpus_path: path of the corpus
		:return: pd.DataFrame
		"""

		df = pd.read_csv(corpus_path, sep=separator, dtype={'User_ID': object})  # , index_col="TweetID"
		if categorize:
			df.User_ID = df.User_ID.astype('category').cat.codes.values
			df.TweetID = df.TweetID.astype('category').cat.codes.values
			df = df[df.User_ID >= 0]

		df['Vector'] = df.apply(lambda row: np.asarray([float(x) for x in row['Vector'][1:-1].split(', ')]), axis=1)

		return df

	@staticmethod
	def corpus_to_sparse_matrix(corpus_path):
		corpus = Parser.parsing_iot_corpus_pandas(corpus_path)

		num_users = corpus.User_ID.max() + 1
		num_tweets = corpus.TweetID.max() + 1

		print(num_users, 'users')
		print(num_tweets, 'tweets')

		# Construct matrix
		mat = sp.dok_matrix((num_users, num_tweets), dtype=np.float32)

		for index, tweet in corpus.iterrows():
			mat[int(tweet.User_ID), int(tweet.TweetID)] = 1.

		return mat

	def get_composant(self, column):
		pass

	def tweet2vec(self, tweet_text):
		sentence_vector = []
		self.load_w2v_model()

		for word in tweet_text:
			try:
				sentence_vector.append(self.model.wv[word])

			except KeyError:
				pass

		# if a tweet word do not appear in the model we put a zeros vector
		if len(sentence_vector) == 0:
			sentence_vector.append(np.zeros_like(self.model.wv["tax"]))

		return np.mean(sentence_vector, axis=0)

	def load_w2v_model(self,
					   path_to_pretrained_model=os.path.join(ROOT_DIR, 'corpus/GoogleNews-vectors-negative300.bin')):
		if self.model is not None:
			return

		self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
		print('GoogleNews-vectors LOADED')

	@staticmethod
	def add_vector_to_corpus(corpus_path, new_corpus_path, write_every=1000):
		"""
		Create a new CleanedText and Vector column on the corpus
		Separate the URLs by space if many
		:param write_every: write in the final file every x lines
		:param corpus_path:
		:param new_corpus_path:
		:return:
		"""
		parser = Parser()
		parser.load_w2v_model()

		corpus = open(corpus_path, 'r', encoding='utf-8')
		new_corpus = open(new_corpus_path, 'w', encoding='utf-8')

		lines = corpus.readlines()
		corpus.close()
		new_lines = []
		last_written = -1
		new_lines.append(lines[0][:-1] + '\tCleanedText\tVector\n')

		for i in range(1, len(lines)):
			parts = lines[i][:-1].split('\t')
			cleaned_tweet = parser.clean_tweet(parts[-2])
			urls = parts[5:-2]
			new_lines.append(
				'\t'.join(parts[:5]) + '\t' +  # TweetID Sentiment TopicID Country Gender
				' '.join(urls) + '\t' +  # URLs separated by space
				'\t'.join(parts[-2:]) + '\t' +  # Text User_ID
				' '.join(cleaned_tweet) + '\t' +  # CleanedText
				str(list(parser.tweet2vec(cleaned_tweet)))  # Vector
				+ '\n')

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
	# Parser.add_vector_to_corpus('corpus/iot-tweets-2009-2016-complet.tsv', 'corpus/iot-tweets-vector.tsv')
	# Parser.add_vector_to_corpus('corpus/iot-tweets-2009-2016-complet.tsv', 'corpus/iot-tweets-vector-new.tsv', write_every=5)

	matrix = Parser.corpus_to_sparse_matrix('corpus/iot-tweets-vector-new.tsv')
	print(matrix)
