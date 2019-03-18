import os

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
		query = DB.get_instance().query(Tweet.vector, Tweet.topic_id)
		if limit is not None:
			query = query.limit(limit)
		for i in query.all():
			X.append(i[0])
			y.append(i[1])

		# self.model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
		# self.model = SVC(gamma='auto')
		# self.model = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')
		self.model = KNeighborsClassifier(n_neighbors=3)
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

		return self.model.predict(vector)

	def compare_classifiers(self):
		X, y = [], []
		query = DB.get_instance().query(Tweet.vector, Tweet.topic_id)
		if self.limit is not None:
			query = query.limit(self.limit)
		for i in query.all():
			X.append(i[0])
			y.append(i[1])

		names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
		         "Random Forest", "AdaBoost", "Naive Bayes"]
		classifiers = [
			KNeighborsClassifier(3),
			SVC(kernel="linear", C=0.025),
			SVC(gamma=2, C=1),
			DecisionTreeClassifier(max_depth=5),
			RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
			AdaBoostClassifier(),
			GaussianNB()]

		f = open("topics_clf_scores.txt", "w+")
		for i in range(len(names)):
			f.write(names[i] + " : " + str(cross_val_score(classifiers[i], X, y, cv=3).mean()))
		f.close()


if __name__ == '__main__':
	clf = TopicsClassifier(limit=10000)
	clf.compare_classifiers()
