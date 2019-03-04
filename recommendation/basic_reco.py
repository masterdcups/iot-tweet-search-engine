import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from definitions import ROOT_DIR
from parser import Parser
from user import User


class BasicReco:

	def __init__(self):
		self.corpus = None
		self.load_corpus()

	def load_corpus(self, corpus_path=os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v31.tsv')):
		if self.corpus is not None:
			return
		self.corpus = Parser.parsing_vector_corpus_pandas(corpus_path)

	def recommended_tweets(self, main_user, k_best=5):
		"""
		Use the cosine simmilarity between the main user thematic vector and all the tweets vectos
		:param main_user: str, name of the user
		:param k_best: number of results to return
		:return: list of recommended tweets
		"""
		user = User(main_user)
		user.load()

		# Add cosine similarity between main_user and all others tweets to DataFrame
		self.corpus['cosine_sim'] = cosine_similarity(np.matrix(self.corpus['Vector'].tolist()),
													  user.vec.reshape(1, -1))

		# Sort the DataFrame by cosine_sim
		self.corpus = self.corpus.sort_values(by=['cosine_sim'], ascending=False)

		# Remove tweets from the main_user
		self.corpus = self.corpus[self.corpus.User_ID != main_user]

		return self.corpus[:k_best]


if __name__ == '__main__':
	reco = BasicReco()
	main_user = '17635797'
	print(reco.recommended_tweets(main_user))
