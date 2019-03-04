import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from db import DB
from models.author import Author


class QueryInfluencerDetection:

	def __init__(self):
		self.topic_vector = None
		self.authors = DB.get_instance().query(Author)
		self.users_topic_vec = [u.topic_vector for u in self.authors]

	def get_influencers(self, topic_vector, percentage_top_user=.5):
		"""
		Return the most central users (influencers) among the X% users most related to the topic
		:param topic_vector: (np.array) vector of topic probabilities
		:param percentage_top_user: (float) percentage of users considered
		:return: a sorted array of
		"""

		cosine_sim = cosine_similarity(self.users_topic_vec, topic_vector.reshape(1, -1))

		results = []
		for i in range(len(cosine_sim)):
			results.append(
				{'user_id': self.authors[i].id, 'sim': cosine_sim[i][0], 'centrality': self.authors[i].centrality})

		results = sorted(results, key=lambda k: k['sim'], reverse=True)
		top_candidates = results[:int(len(results) * percentage_top_user)]
		return [x['user_id'] for x in sorted(top_candidates, key=lambda k: k['centrality'], reverse=True)]


if __name__ == '__main__':
	qid = QueryInfluencerDetection()
	influencers = qid.get_influencers(np.asarray([0.25, 0.25, 0.25, 0.25]))
	print(influencers)
