import numpy as np
from model_prediction import ModelPrediction


class PredictionProfile:
    """Class to predict users' sentiment, gender and country"""

    def __init__(self):
        """this class needs a dictionary with TweetID, Sentiment, TopicID, Country, Gender, URLs, Text, Vector"""
        self.model_prediction = ModelPrediction()

    def gender_prediction(self, vector):
        """Method to predict user's gender using a SVM classifier"""

        return self.model_prediction.gender_model().predict(vector.reshape(1, -1))[0]

    def sentiment_prediction(self, vector):
        """Method to predict user's sentiment using a CNN"""

        return self.model_prediction.sentiment_model().predict(vector.reshape(1, -1))[0]

    def country_prediction(self, vector):
        """Method to predict user's country using a Naive Bayes network"""
        return self.model_prediction.country_model().predict(vector.reshape(1, -1))[0]


if __name__ == '__main__':
    pred = PredictionProfile()

    print(pred.gender_prediction(np.zeros(300)))
    print(pred.sentiment_prediction(np.zeros(300)))
    print(pred.country_prediction(np.zeros(300)))
