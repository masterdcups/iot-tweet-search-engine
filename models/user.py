import numpy as np
from sqlalchemy import Column, Text, Integer, DateTime, ARRAY, Float, func
from sqlalchemy.ext.declarative import declarative_base

from db import DB
from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier

Base = declarative_base()


class User(Base):
	__tablename__ = 'users'

	id = Column(Integer, primary_key=True)
	date_created = Column(DateTime, default=func.current_timestamp())
	date_modified = Column(DateTime, default=func.current_timestamp(),
	                       onupdate=func.current_timestamp())
	vector = Column(ARRAY(Float), nullable=True, unique=False)
	nb_click = Column(Integer, nullable=True, unique=False)
	localisation = Column(Text, nullable=True, unique=False)
	gender = Column(Text, nullable=True, unique=False)
	emotion = Column(Text, nullable=True, unique=False)
	topic_vector = Column(ARRAY(Float), nullable=True, unique=False)

	@staticmethod
	def load(user_id):
		u = DB.get_instance().query(User).filter_by(id=user_id).first()
		if u is None:
			u = User()
			u.nb_click = 0
			u.vector = np.zeros(300)
			u.gender = 'andy'
			DB.get_instance().add(u)
		return u

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

		tpc = TopicsClassifier()
		pp = PredictionProfile()
		self.predict_profile(tpc, pp)

		DB.get_instance().commit()
