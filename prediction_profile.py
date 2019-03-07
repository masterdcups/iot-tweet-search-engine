from db import DB
from models.tweet import Tweet

from model_prediction import ModelPrediction


class PredictionProfile:
	"""Class to predict users' sentiment, gender and country"""

	def __init__(self, limit: int = None):
		"""this class needs a dictionary with TweetID, Sentiment, TopicID, Country, Gender, URLs, Text, Vector"""

		iterator = PredictionProfile._iterator(limit=limit)
		self.model_prediction = ModelPrediction(iterator)

	def gender_prediction(self, vector):
		"""Method to predict user's gender using a SVM classifier"""

		return self.model_prediction.gender_model().predict(vector.reshape(1, -1))[0]

	def sentiment_prediction(self, vector):
		"""Method to predict user's sentiment using a CNN"""

		return self.model_prediction.sentiment_model().predict(vector.reshape(1, -1))[0]

	def country_prediction(self, vector):
		"""Method to predict user's country using a Naive Bayes network"""
		return self.model_prediction.country_model().predict(vector.reshape(1, -1))[0]

	@staticmethod
	def _iterator(limit: int = None):
		query = DB.get_instance().query(Tweet.vector, Tweet.sentiment, Tweet.gender, Tweet.country)
		if limit is not None:
			query = query.limit(limit)

		for t in query.all():
			yield t


if __name__ == '__main__':
	pred = PredictionProfile(limit=1000)

	pred.model_prediction.tweak_hyperparameters("SVM")
# pred.model_prediction.tweak_hyperparameters("MLP")
#  print(pred.gender_prediction(np.zeros(300)))
#  print(pred.sentiment_prediction(np.zeros(300)))
#  print(pred.country_prediction(np.zeros(300)))
