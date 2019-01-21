# give the vector topic of a query

import os

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from parser import Parser


class TopicsClassifier:
	model_name = 'topic_model.joblib'

	def __init__(self, dir_path='saved_models'):
		self.model = None

		if not os.path.exists(dir_path):
			os.mkdir(dir_path)
		self.model_path = os.path.join(dir_path, TopicsClassifier.model_name)

	def train(self):
		# todo : to change with the real corpus
		corpus = Parser.parsing_iot_corpus("corpus/fake-iot-corpus.tsv")
		print('Corpus loaded')

		X = [t['Vector'] for t in corpus]
		y = [t['TopicID'] for t in corpus]

		self.model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
		# self.model = SVC(gamma='auto')
		# self.model = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')
		# self.model = KNeighborsClassifier(n_neighbors=3)
		self.model.fit(X, y)  # probability = True

	def save(self):
		assert (self.model is not None)

		dump(self.model, self.model_path)

	def load(self):
		self.model = load(self.model_path)

	def predict(self, vector):
		if not os.path.exists(self.model_path):
			self.train()
			self.save()
		else:
			self.load()

		return self.model.predict_proba(vector)


if __name__ == '__main__':
	print('TopicClassifier')
	clf = TopicsClassifier()
	print(clf.predict(np.zeros(300).reshape(1, -1)))
