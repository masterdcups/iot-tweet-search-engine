import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from db import DB
from definitions import ROOT_DIR
from models.tweet import Tweet
from parser import Parser


class BasicReco:

	def __init__(self):
		self.corpus = None
		# self.load_corpus()
		self.parser = Parser()
		self.tweets = DB.get_instance().query(Tweet).limit(100)
		print('tweets loaded')

	def load_corpus(self, corpus_path=os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v31.tsv')):
		if self.corpus is not None:
			return
		self.corpus = Parser.parsing_vector_corpus_pandas(corpus_path)

	def recommended_tweets(self, main_user, k_best=5):
		"""
		Use the cosine similarity between the main user thematic vector and all the tweets vectors
		:param main_user: str, name of the user
		:param k_best: number of results to return
		:return: list of recommended tweets
		"""

		# Add cosine similarity between main_user and all others tweets to DataFrame
		cosine_sim = cosine_similarity(np.matrix([np.array(t.vector) for t in self.tweets]),
									   np.array(main_user.vector).reshape(1, -1))

		results = []
		for i in range(len(cosine_sim)):
			results.append({'cosine_sim': cosine_sim[i], 'tweet': self.tweets[i]})

		# Sort the DataFrame by cosine_sim
		results = sorted(results, key=lambda k: k['cosine_sim'], reverse=True)
		# self.corpus = self.corpus.sort_values(by=['cosine_sim'], ascending=False)

		# Remove tweets from the main_user
		results = list(filter(lambda x: x['tweet'].user_id != main_user.id, results))

		return [t['tweet'] for t in results[:k_best]]


if __name__ == '__main__':
	reco = BasicReco()
	main_user = '4'
	print(reco.recommended_tweets(main_user))
