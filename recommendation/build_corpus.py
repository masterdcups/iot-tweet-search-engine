import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from db import DB
from definitions import ROOT_DIR
from models.tweet import Tweet


class BuildCorpus:

	def __init__(self, limit=None, num_negatives=2):
		self.limit = limit
		self.num_negatives = num_negatives

	def call(self):
		query = DB.get_instance().query(Tweet.user_id, Tweet.user_name, Tweet.id)

		if self.limit is not None:
			query = query.limit(self.limit)

		original_corpus = pd.read_sql(query.statement, query.session.bind, coerce_float=False)
		original_corpus['user_id'] = original_corpus.user_id.astype('category').cat.codes.values
		original_corpus['id'] = original_corpus.id.astype('category').cat.codes.values

		print('tweets loaded')

		# num_users = len(original_corpus.user_name.unique())
		# num_tweets = len(original_corpus.id.unique())

		corpus = original_corpus[['user_id', 'id']]

		# like_rt_file = open(like_rt_graph, 'r')
		# for line in like_rt_file:
		# 	parts = line[:-1].split(' ')
		# 	user_name = parts[0]
		# 	user_id = original_corpus[original_corpus.user_name == user_name].User_ID_u.iloc[0]
		# 	tweets = parts[1:]  # like or RT tweets
		#
		# 	for tweet in tweets:
		# 		if len(original_corpus[original_corpus.id == int(tweet)]) > 0:
		# 			tweet_id = original_corpus[original_corpus.id == int(tweet)].TweetID_u.iloc[0]
		#
		# 			corpus = corpus.append({'User_ID_u': user_id, 'TweetID_u': tweet_id}, ignore_index=True)
		#
		# like_rt_file.close()

		corpus['rating'] = 1
		tweets = set(corpus.id)

		i = 0

		for index, line in corpus.iterrows():
			u = line.user_id

			if i % 100 == 0:
				print(i)
			i += 1

			user_tweets = set(corpus[(corpus.user_id == u) & (corpus.rating == 1)].id)
			tweets_to_choose = random.sample(list(tweets.difference(user_tweets)), self.num_negatives)

			for t in tweets_to_choose:
				corpus = corpus.append({'user_id': u, 'id': t, 'rating': 0}, ignore_index=True)

		train_corpus, test_corpus = train_test_split(corpus, test_size=0.2)

		train_corpus.to_csv(os.path.join(ROOT_DIR, 'corpus/model_reco/train_corpus.csv'), index=False)
		test_corpus.to_csv(os.path.join(ROOT_DIR, 'corpus/model_reco/test_corpus.csv'), index=False)
