import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from db import DB
from models.author import Author
from models.favorite import Favorite
from models.tweet import Tweet
from profile_oneHotEncoder import ProfileOneHotEncoder


class UserReco:

	def __init__(self, user):
		self.user = user.id
		self.user_vec = ProfileOneHotEncoder.add_info_to_vec(self.user.vector, self.user.gender, self.user.location,
		                                                     self.user.emotion)
		self.graph = Author.load_graph()
		self.graph.add_node(user.id)
		favorite_tweets = DB.get_instance().query(Favorite.tweet_id).filter(Favorite.user_id == user.id)
		self.authors_liked = DB.get_instance().query(Tweet.user_name).filter(
			Tweet.id.in_(favorite_tweets.subquery())).all()
		for a in self.authors_liked:
			self.graph.add_edge(user.id, a[0])

	def rerank_authors(self, authors_prio):
		"""
		rerank results from authors_prio based on their similarity with the user
		:param authors_prio: the jaccard coefficient for the link prediction between the user and each kept author
		:return:
		"""
		reranked_reco = []
		for a, p in authors_prio:
			author = DB.get_instance().query(Author).filter(Author.name == a).first()
			author_vec = ProfileOneHotEncoder.add_info_to_vec(author.vector, author.gender, author.localisation,
			                                                  author.emotion).reshape(1, -1)
			sim = cosine_similarity(self.user_vec, author_vec)
			reranked_reco.append({'author_name': a, 'sim': sim[0]})
		return sorted(reranked_reco, key=lambda k: k['sim'], reverse=True)

	def users_to_recommend(self, nb_reco_user=5):
		"""
		compute the authors to recommend to the user based on link prediction and similarity
		:param nb_reco_user: number of users to recommend
		:return:
		"""
		ebunch = []
		authors = set(self.graph.nodes())
		authors.remove(self.user.id)
		for a in self.authors_liked:
			authors.remove(a)
		for author in authors:
			ebunch.append((self.user.id, author))

		preds = nx.jaccard_coefficient(self.graph, ebunch)
		reco_prio = []
		for u, a, p in preds:
			reco_prio.append({'author_name': a, 'prio': p})

		reco_prio = sorted(reco_prio, key=lambda k: k['prio'], reverse=True)[:nb_reco_user]
		reco_prio = self.rerank_authors(reco_prio)

		return [x['author_name'] for x in reco_prio]
