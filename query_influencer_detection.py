import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from user import User


class QueryInfluencerDetection:

	def __init__(self):
		return

	@staticmethod
	def select_influencer(topic_vector, nb_user=1):
		users = User.get_all_users()

		users_topic_vec = [u.topic_vector for u in users]
		cosine_sim = cosine_similarity(users_topic_vec, topic_vector.reshape(1, -1))

		users_id = [u.id for u in users]
		results = []
		for i in range(len(cosine_sim)):
			results.append({'user_id': users_id[i], 'sim': cosine_sim[i][0]})

		results = sorted(results, key=lambda k: k['sim'], reverse=True)
		return [x['user_id'] for x in results[:nb_user]]


if __name__ == '__main__':
	influencers = QueryInfluencerDetection.select_influencer(np.asarray([0.25, 0.25, 0.25, 0.25]))
	print(influencers)
