import os

import networkx as nx
import numpy as np
from sqlalchemy import Column, Text, Integer, ARRAY, Float

from db import DB
from definitions import ROOT_DIR
from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier


class Author(DB.get_base()):
	__tablename__ = 'authors'

	id = Column(Integer, primary_key=True)
	vector = Column(ARRAY(Float), nullable=True, unique=False)
	nb_click = Column(Integer, nullable=True, unique=False)
	localisation = Column(Text, nullable=True, unique=False)
	gender = Column(Text, nullable=True, unique=False)
	emotion = Column(Text, nullable=True, unique=False)
	topic = Column(Integer, nullable=True, unique=False)
	centrality = Column(Float, nullable=True, unique=False)
	name = Column(Text, nullable=True, unique=False)

	@staticmethod
	def load(user_id=None, user_name=None):
		assert user_id is not None or user_name is not None

		u = None
		if user_id is not None:
			u = DB.get_instance().query(Author).filter_by(id=user_id).first()
		elif user_name is not None:
			u = DB.get_instance().query(Author).filter_by(name=user_name).first()

		return u

	def predict_profile(self, topics_classifier, prediction_profile):
		"""
		Call all the predictions models to fill the localisation, gender, etc
		:return:
		"""

		self.localisation = prediction_profile.country_prediction(np.array(self.vector))
		self.gender = prediction_profile.gender_prediction(np.array(self.vector))
		self.emotion = prediction_profile.sentiment_prediction(np.array(self.vector))
		self.topic = int(topics_classifier.predict(np.array(self.vector).reshape(1, -1))[0])

	def update_profile(self, vec):
		"""
		Update the profile of the user with the new vec param
		:param vec: (np.array) vector of the tweet to add
		:return:
		"""
		self.nb_click += 1
		for i in range(len(self.vector)):
			self.vector[i] = (self.vector[i] * (self.nb_click - 1)) / self.nb_click + (vec[i] / self.nb_click)

	@staticmethod
	def create_authors(limit=None):
		from models.tweet import Tweet
		tpc = TopicsClassifier(limit=limit)
		pp = PredictionProfile(limit=limit)

		query = DB.get_instance().query(Tweet.user_id, Tweet.vector, Tweet.user_name)

		if limit is not None:
			query = query.limit(limit)

		for t in query.all():
			a = Author.load(user_name=t.user_name)
			if a is None:
				a = Author()
				a.id = t.user_id
				a.name = t.user_name
				a.nb_click = 0
				a.vector = np.zeros(300)
				DB.get_instance().add(a)
			a.update_profile(np.array(t.vector))
			DB.get_instance().commit()

		print('Authors created')

		# graph = Author.load_graph()
		# centralities = nx.eigenvector_centrality(graph)
		for author in DB.get_instance().query(Author).all():
			# author.centrality = centralities[author.name] if author.name in centralities else 0.
			author.predict_profile(tpc, pp)
			DB.get_instance().commit()

	@staticmethod
	def load_graph(filename=os.path.join(ROOT_DIR, 'corpus/followers_matrix.tsv')):
		return nx.DiGraph(nx.read_adjlist(filename))


if __name__ == '__main__':
	Author.create_authors(limit=10)
