import os

import pandas as pd
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from db import DB
from definitions import ROOT_DIR
from models.tweet import Tweet


class TopicsClassifier:
	model_name = 'topic_model.joblib'

	def __init__(self, dir_path=os.path.join(ROOT_DIR, 'saved_models'), pd_corpus=None, limit=None):
		"""
		:type pd_corpus: pandas.DataFrame
		"""
		self.model = None
		self.corpus = pd_corpus
		self.limit = limit

		if not os.path.exists(dir_path):
			os.mkdir(dir_path)
		self.model_path = os.path.join(dir_path, TopicsClassifier.model_name)

	def train(self, limit: int = None):

		X, y = [], []
		query = DB.get_instance().query(Tweet.vector, Tweet.topic_id).filter('topic_id is not null')
		if limit is not None:
			query = query.limit(limit)
		for i in query.all():
			X.append(i[0])
			y.append(i[1])

		# self.model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
		# self.model = SVC(gamma='auto')
		# self.model = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')
		# self.model = KNeighborsClassifier(n_neighbors=3)
		self.model.fit(X, y)  # probability = True

	def save(self):
		assert (self.model is not None)

		dump(self.model, self.model_path)

	def load(self):
		if self.model is not None:
			return

		self.model = load(self.model_path)

	def predict(self, vector):
		if not os.path.exists(self.model_path):
			self.train(limit=self.limit)
			self.save()
		else:
			self.load()

		return self.model.predict_proba(vector)

	def tweak_hyperparameters(self):
		X, y = [], []
		query = DB.get_instance().query(Tweet.vector, Tweet.topic_id).filter('topic_id is not null')
		for i in query.all():
			X.append(i[0])
			y.append(i[1])

		model = KNeighborsClassifier()
		param_grid = {
			'n_neighbors': [2, 3, 5, 8, 10, 15, 20],
			'algorithm': ['auto']
		}

		clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
		clf.fit(X, y)

		df = pd.DataFrame(clf.cv_results_)
		df.to_csv(r'' + 'k_neighbors.txt', index=None, sep=' ')


if __name__ == '__main__':
	clf = TopicsClassifier()
	clf.tweak_hyperparameters()
# clf.train(limit=100000)
# clf.save()
# print(clf.predict(np.zeros(300).reshape(1, -1)))
