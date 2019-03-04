import os

import gensim
import nltk
import numpy as np
import pandas as pd
import preprocessor
import scipy.sparse as sp
from spellchecker import SpellChecker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from definitions import ROOT_DIR
from models.tweet import Tweet


class Parser:
	ENGINE_ADDR = 'postgresql+psycopg2://postgres:password@localhost:5432/iot_tweet'  # 'postgresql+psycopg2://postgres:password@/iot_tweet?host=/cloudsql/iot-tweet:europe-west3:main-instance'

	def __init__(self):
		self.load_nltk()
		self.model = None
		self.abbreviations = None
		self.spell_check = None
		self.session = None

		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.MENTION, preprocessor.OPT.RESERVED,
								 preprocessor.OPT.EMOJI, preprocessor.OPT.SMILEY)

		self.load_db_tweets()

	def load_db_tweets(self):
		engine = create_engine(Parser.ENGINE_ADDR, echo=True)
		Session = sessionmaker(bind=engine)
		self.session = Session()

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

		file_name = os.path.join(ROOT_DIR, "corpus/slang.txt")
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

	def get_vector(self, tweet_id, as_np_array=False):
		"""
		Return the vector of a specific tweet
		:param tweet_id: id of the tweet
		:param as_np_array: convert the vector into numpy.array
		:return: vector: list or numpy.array
		"""
		if self.session.query(Tweet.vector).filter_by(id=int(tweet_id)).first() is None:
			return None
		vector = self.session.query(Tweet.vector).filter_by(id=int(tweet_id)).first()[0]

		if as_np_array:
			vector = np.array(vector)

		return vector

	def get_all_vectors(self, tweet_ids=None, limit=None):
		"""
		Return all the vectors
		:param tweet_ids: is specified, filter the vectors to return with tweet_id
		:param limit: nb of results
		:return: dict tweet_id -> vector (list)
		"""
		query = self.session.query(Tweet.id, Tweet.vector)

		if tweet_ids is not None:
			query = query.filter(Tweet.id.in_(tweet_ids))
		if limit is not None:
			query = query.limit(limit)

		return dict(query.all())


	@staticmethod
	def parsing_vector_corpus_pandas(corpus_path, separator='\t', categorize=False, vector_asarray=True):
		"""
		Parse the corpus and return a Pandas DataFrame
		:param categorize: boolean to make the tweet and user ids start to 0
		:param separator:
		:param corpus_path: path of the corpus
		:return: pandas.DataFrame
		"""

		df = pd.read_csv(corpus_path, sep=separator, dtype={'User_ID': object})  # , index_col="TweetID"
		df = df.dropna(subset=['User_ID'])  # remove tweets without users
		if categorize:
			df['User_ID_u'] = df.User_ID.astype('category').cat.codes.values
			df['TweetID_u'] = df.TweetID.astype('category').cat.codes.values
			df = df[df.User_ID_u >= 0]

		# This takes 20 seconds
		if vector_asarray:
			df['Vector'] = df.apply(lambda row: np.asarray([float(x) for x in row['Vector'][1:-1].split(', ')]), axis=1)

		return df

	@staticmethod
	def parsing_base_corpus_pandas(corpus_path, separator='\t', categorize=False):
		"""
		Parse the corpus and return a Pandas DataFrame
		:param categorize: boolean to make the tweet and user ids start to 0
		:param separator:
		:param corpus_path: path of the corpus
		:return: pandas.DataFrame
		"""

		df = pd.read_csv(corpus_path, sep=separator, dtype={'User_ID': object})  # , index_col="TweetID"
		df = df.dropna(subset=['User_ID'])  # remove tweets without users

		if categorize:
			df['User_ID_u'] = df.User_ID.astype('category').cat.codes.values
			df['TweetID_u'] = df.TweetID.astype('category').cat.codes.values
			df = df[df.User_ID_u >= 0]

		return df

	@staticmethod
	def corpus_to_sparse_matrix(corpus_path):
		corpus = Parser.parsing_vector_corpus_pandas(corpus_path)

		num_users = corpus.User_ID.max() + 1
		num_tweets = corpus.TweetID.max() + 1

		print(num_users, 'users')
		print(num_tweets, 'tweets')

		# Construct matrix
		mat = sp.dok_matrix((num_users, num_tweets), dtype=np.float32)

		for index, tweet in corpus.iterrows():
			mat[int(tweet.User_ID), int(tweet.TweetID)] = 1.

		return mat

	@staticmethod
	def vector_string_to_array(vector):
		return np.asarray([float(x) for x in vector[1:-1].split(', ')])

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

		return np.mean(sentence_vector, axis=0, dtype=float)

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
			cleaned_tweet = parser.clean_tweet(parts[-6])
			urls = parts[5:-6]
			new_lines.append(
				'\t'.join(parts[:5]) + '\t' +  # TweetID Sentiment TopicID Country Gender
				' '.join(urls) + '\t' +  # URLs separated by space
				parts[-6] + '\t' +  # Text
				parts[-5] + '\t' +  # User_ID
				parts[-4][1:-1] + '\t' +  # User_Name without quotes
				parts[-3][1:-1] + '\t' +  # Date without quotes
				'\t'.join(parts[-2:]) + '\t' +  # Hashtags Indication
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
	p = Parser()
	print(p.get_all_vectors(limit=20))
	exit()
	vector = p.get_vector(80434341692663808089, as_np_array=True)
	print(vector)
