import os

from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from definitions import ROOT_DIR
from parser import Parser


class ModelPrediction:
	path_gend_mod = 'gender_model.joblib'
	path_sent_mod = 'sentiment_model.joblib'
	path_coun_mod = 'country_model.joblib'

	def __init__(self, corpus=None, dir_path=os.path.join(ROOT_DIR, 'saved_models')):
		"""

		:type corpus: pandas.DataFrame
		"""
		self.dir_path = dir_path
		self.tweets = corpus

		self.gender_mod = None
		self.sentiment_mod = None
		self.country_mod = None

		if not os.path.exists(dir_path):
			os.mkdir(dir_path)

	def load_corpus(self):
		if self.tweets is None:
			self.tweets = Parser.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, "corpus/iot-tweets-vector-v3.tsv"))

	def gender_model(self, file_path=path_gend_mod):
		"""Method to create model prediction user's gender using a SVM classifier"""
		full_path = os.path.join(self.dir_path, file_path)

		if not os.path.exists(full_path):

			self.load_corpus()

			X = self.tweets.Vector.tolist()
			y = self.tweets.Gender.tolist()

			self.gender_mod = svm.SVC(kernel='linear')
			self.gender_mod.fit(X, y)

			dump(self.gender_mod, full_path)
		else:
			if self.gender_mod is None:
				self.gender_mod = load(full_path)

		return self.gender_mod

	def sentiment_model(self, file_path=path_sent_mod):
		"""Method to create model prediction user's sentiment using a CNN"""
		full_path = os.path.join(self.dir_path, file_path)

		if not os.path.exists(full_path):

			self.load_corpus()

			X = self.tweets.Vector.tolist()
			y = self.tweets.Sentiment.tolist()

			self.sentiment_mod = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
			self.sentiment_mod.fit(X, y)

			dump(self.sentiment_mod, full_path)
		else:
			if self.sentiment_mod is None:
				self.sentiment_mod = load(full_path)

		return self.sentiment_mod

	def country_model(self, file_path=path_coun_mod):
		"""Method to create model prediction user's country using a Naive Bayes network"""
		full_path = os.path.join(self.dir_path, file_path)

		if not os.path.exists(full_path):

			self.load_corpus()

			X = self.tweets.Vector.tolist()
			y = self.tweets.Country.tolist()

			self.country_mod = GaussianNB()
			self.country_mod.fit(X, y)

			dump(self.country_mod, full_path)
		else:
			if self.country_mod is None:
				self.country_mod = load(full_path)

		return self.country_mod

	def tweak_hyperparameters(self, type_model):
		self.load_corpus()

		if type_model == "SVM":
			X = self.tweets.Vector.tolist()
			y = self.tweets.Gender.tolist()

			model = svm.SVC()
			param_grid = [
				{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
				{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
			]
			clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
			clf.fit(X, y)

			# Best parameter set
			print('Best parameters found:\n', clf.best_params_)
		else:
			X = self.tweets.Vector.tolist()
			y = self.tweets.Sentiment.tolist()

			model = MLPClassifier(max_iter=100)
			param_grid = {
				'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
				'activation': ['tanh', 'relu'],
				'solver': ['sgd', 'adam'],
				'alpha': [0.0001, 0.05],
				'learning_rate': ['constant', 'adaptive'],
			}
			clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
			clf.fit(X, y)

			# Best parameter set
			print('Best parameters found:\n', clf.best_params_)


if __name__ == '__main__':
	corpus = Parser.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v3.tsv'))
	model = ModelPrediction(corpus=corpus)
	model.tweak_hyperparameters("SVM")
	model.tweak_hyperparameters("MLP")
	# print(model.gender_model())
	# print(model.sentiment_model())
	# print(model.country_model())
	# print(model.gender_model().predict(np.zeros(300).reshape(1, -1)))
	# print(model.sentiment_model().predict(np.zeros(300).reshape(1, -1)))
	# print(model.country_model().predict(np.zeros(300).reshape(1, -1)))
