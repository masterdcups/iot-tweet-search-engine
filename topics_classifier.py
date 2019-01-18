import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from parser import Parser


class TopicsClassifier:

	def __init__(self):
		self.model = None

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

	def save(self, path):
		assert (self.model is not None)

		dump(self.model, path)

	def load(self, path):
		self.model = load(path)

	def predict(self, vector):
		# todo : save or load model !
		if self.model is None:
			self.train()

		return self.model.predict_proba(vector)


if __name__ == '__main__':
	print('TopicClassifier')
	clf = TopicsClassifier()
	print(clf.predict(np.zeros(300).reshape(1, -1)))
