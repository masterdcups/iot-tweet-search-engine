import re

import gensim
import nltk
import numpy as np
import preprocessor
from spellchecker import SpellChecker


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
		self.abbreviations = [line[:-1].split('=') for line in file.readlines()]
		file.close()

	# with open(file_name, 'r') as myCSVfile:
	# 	# Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
	# 	self.abbreviations = csv.reader(myCSVfile, delimiter="=")
	#
	# myCSVfile.close()

	def replace_abbreviations(self, tokens):
		"""
		Replace the abbreviations (OMG -> Oh My God) based on the dictionary in slang.txt
		:param tokens: words of the tweet
		:return: words with abbreviations replaced by their meaning
		"""

		self.load_abbreviations()

		for i in range(len(tokens)):
			# Removing Special Characters.
			_token = re.sub('[^a-zA-Z0-9-_.]', '', tokens[i])
			for row in self.abbreviations:
				# Check if selected word matches short forms[LHS] in text file.
				if tokens[i].upper() == row[0]:
					# If match found replace it with its Abbreviation in text file.
					tokens[i] = row[1]

		return tokens

	def remove_stopwords_spelling_mistakes(self, tokens):
		"""
		Remove stopwords and corrects spelling mistakes
		:param spell: Object to correct spelling mistakes
		:param tokens: words of the tweet
		:return: words cleaned and corrected
		"""

		self.load_spell_check()

		clean_tokens = []
		for token in tokens:
			# correction of spelling mistakes
			token = self.spell_check.correction(token)
			if token not in nltk.corpus.stopwords.words('english'):
				clean_tokens.append(token)
		return clean_tokens

	@staticmethod
	def parsing_iot_corpus(corpus_path, clean_tweet=True):
		"""
		Parse the corpus and return the list of tweets with characteristics
		:param clean_tweet: boolean clean the text of the tweets with nltk
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
				tweet_infos['Text'] = parser.clean_tweet(tweet[-3]) if clean_tweet else tweet[-3]
				tweet_infos['Author'] = tweet[-2]
				tweet_infos['Vector'] = np.asarray([float(x) for x in tweet[-1][1:-1].split(', ')])
				tweets.append(tweet_infos)

		file.close()
		return tweets

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

	def load_w2v_model(self):
		if self.model is not None:
			return

		self.model = gensim.models.KeyedVectors.load_word2vec_format('corpus/GoogleNews-vectors-negative300.bin',
																	 binary=True)

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
		parser.load_w2v_model()

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
				list(parser.tweet2vec(parser.clean_tweet(lines[i].split('\t')[-2])))) + '\n')

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
