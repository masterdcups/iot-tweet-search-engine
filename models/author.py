import os

import networkx as nx
import numpy as np
from sqlalchemy import Column, Text, Integer, ARRAY, Float
from sqlalchemy.ext.declarative import declarative_base

from db import DB
from definitions import ROOT_DIR
from models.tweet import Tweet
from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier

Base = declarative_base()


class Author(Base):
	__tablename__ = 'authors'

	id = Column(Integer, primary_key=True)
	vector = Column(ARRAY(Float), nullable=True, unique=False)
	nb_click = Column(Integer, nullable=True, unique=False)
	localisation = Column(Text, nullable=True, unique=False)
	gender = Column(Text, nullable=True, unique=False)
	emotion = Column(Text, nullable=True, unique=False)
	topic_vector = Column(ARRAY(Float), nullable=True, unique=False)
	centrality = Column(Float, nullable=True, unique=False)
	name = Column(Text, nullable=True, unique=False)

	@staticmethod
	def load(author_id):
		return DB.get_instance().query(Author).filter_by(id=author_id).first()

	def predict_profile(self, topics_classifier, prediction_profile):
		"""
		Call all the predictions models to fill the localisation, gender, etc
		:return:
		"""

		self.localisation = prediction_profile.country_prediction(self.vector)
		self.gender = prediction_profile.gender_prediction(self.vector)
		self.emotion = prediction_profile.sentiment_prediction(self.vector)
		self.topic_vector = topics_classifier.predict(self.vector.reshape(1, -1))[0]

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
		tpc = TopicsClassifier(limit=limit)
		pp = PredictionProfile(limit=limit)

		query = DB.get_instance().query(Tweet.user_id, Tweet.vector, Tweet.user_name).filter(
			'user_id is not null')

		if limit is not None:
			query = query.limit(limit)

		for t in query.all():
			a = Author.load(t.user_id)
			if a is None:
				a = Author()
				a.id = t.user_id
				a.name = t.user_name
				a.nb_click = 0
				a.vector = np.zeros(300)
				DB.get_instance().add(a)
			a.update_profile(np.array(t.vector))

		graph = Author.load_graph()
		centralities = nx.eigenvector_centrality(graph)
		for author in DB.get_instance().query(Author).all():
			author.centrality = centralities[author.name] if author.name in centralities else 0.
			author.predict_profile(tpc, pp)
		DB.get_instance().commit()

	@staticmethod
	def load_graph(filename=os.path.join(ROOT_DIR, 'corpus/followers_matrix.tsv')):
		return nx.DiGraph(nx.read_adjlist(filename))


if __name__ == '__main__':
	Author.create_authors(limit=50)
