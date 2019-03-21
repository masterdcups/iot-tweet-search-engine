from db import DB
from models.author import Author


class QueryInfluencerDetection:

	@staticmethod
	def get_influencers(topic, nb_users=5):
		"""
		Return the X most central authors (influencers) related to the topic
		:param topic: (int) topic of the user
		:param nb_users: (int) percentage of users considered
		:return: a sorted array of authors based on their centrality
		"""
		return DB.get_instance().query(Author).filter(Author.topic == topic).order_by(
			Author.centrality).limit(nb_users)


if __name__ == '__main__':

	for a in QueryInfluencerDetection.get_influencers(1):
		print(a.name)
