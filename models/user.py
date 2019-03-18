import numpy as np
from sqlalchemy import Column, Text, Integer, DateTime, ARRAY, Float, func, String
from sqlalchemy.orm import relationship

from db import DB
from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier


class User(DB.get_base()):
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
	topic = Column(Integer, nullable=True, unique=False)
	username = Column(String, nullable=False, unique=True)
	password = Column(String, nullable=False, unique=False)

	favs = relationship("Favorite", back_populates='user')

	@staticmethod
	def load(user_id=None, user_name=None):
		assert user_id is not None or user_name is not None

		u = None
		if user_id is not None:
			u = DB.get_instance().query(User).filter_by(id=user_id).first()
		elif user_name is not None:
			u = DB.get_instance().query(User).filter_by(username=user_name).first()

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

	def remove_favorite(self, tweet):

		from models.favorite import Favorite
		from models.tweet import Tweet

		DB.get_instance().query(Favorite).filter_by(user_id=self.id).filter_by(tweet_id=tweet.id).delete(
			synchronize_session='fetch')

		favs = DB.get_instance().query(Favorite.tweet_id, Tweet.vector).join(Tweet,
																			 Tweet.id == Favorite.tweet_id).filter(
			'favorites.user_id=' + str(self.id)).all()
		vects = [f[1] for f in favs]
		self.vector = np.mean(vects, axis=0)

		tpc = TopicsClassifier()
		pp = PredictionProfile()
		self.predict_profile(tpc, pp)

		DB.get_instance().commit()

	def update_profile(self, vec):
		"""
		Update the profile of the user with the new vec param
		:param vec: (np.array) vector of the tweet to add
		:return:
		"""

		assert type(vec) == np.array

		self.nb_click += 1
		for i in range(len(self.vector)):
			self.vector[i] = (self.vector[i] * (self.nb_click - 1)) / self.nb_click + (vec[i] / self.nb_click)

		tpc = TopicsClassifier()
		pp = PredictionProfile()
		self.predict_profile(tpc, pp)

		DB.get_instance().commit()
