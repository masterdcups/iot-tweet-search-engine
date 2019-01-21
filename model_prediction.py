# create and save model to predict gender, sentiment and country of an user

import os

from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from parser import Parser


class ModelPrediction:
	path_gend_mod = 'gender_model.joblib'
	path_sent_mod = 'sentiment_model.joblib'
	path_coun_mod = 'country_model.joblib'

	def __init__(self, corpus_path, dir_path='saved_models'):
		self.dir_path = dir_path
		self.tweets = None
		self.corpus_path = corpus_path

		if not os.path.exists(dir_path):
			os.mkdir(dir_path)

	def load_corpus(self):
		if self.tweets is None:
			self.tweets = Parser.parsing_iot_corpus(self.corpus_path)

	def gender_model(self, file_path=path_gend_mod):
		"""Method to create model prediction user's gender using a SVM classifier"""
		full_path = os.path.join(self.dir_path, file_path)

		if not os.path.exists(full_path):

			self.load_corpus()

			X = [t['Vector'] for t in self.tweets]
			y = [t['Gender'] for t in self.tweets]

			model = svm.SVC(kernel='linear')
			model.fit(X, y)

			dump(model, full_path)
		else:
			model = load(full_path)

		return model

	def sentiment_model(self, file_path=path_sent_mod):
		"""Method to create model prediction user's sentiment using a CNN"""
		full_path = os.path.join(self.dir_path, file_path)

		if not os.path.exists(full_path):

			self.load_corpus()

			X = [t['Vector'] for t in self.tweets]
			y = [t['Sentiment'] for t in self.tweets]

			clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
			clf.fit(X, y)

			dump(clf, full_path)
		else:
			clf = load(full_path)

		return clf

	def country_model(self, file_path=path_coun_mod):
		"""Method to create model prediction user's country using a Naive Bayes network"""
		full_path = os.path.join(self.dir_path, file_path)

		if not os.path.exists(full_path):

			self.load_corpus()

			X = [t['Vector'] for t in self.tweets]
			y = [t['Country'] for t in self.tweets]

			gnb = GaussianNB()
			gnb.fit(X, y)

			dump(gnb, full_path)
		else:
			gnb = load(full_path)

		return gnb

	def tweak_hyperparameters(self, type_model):
		self.load_corpus()

		if type_model == "SVM":
			X = [t['Vector'] for t in self.tweets]
			y = [t['Gender'] for t in self.tweets]

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
			X = [t['Vector'] for t in self.tweets]
			y = [t['Sentiment'] for t in self.tweets]

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
	model = ModelPrediction("corpus/fake-iot-corpus.tsv")
	model.tweak_hyperparameters("SVM")
	model.tweak_hyperparameters("MLP")
# print(model.gender_model())
# print(model.sentiment_model())
# print(model.country_model())
# print(model.gender_model().predict(np.zeros(300).reshape(1, -1)))
# print(model.sentiment_model().predict(np.zeros(300).reshape(1, -1)))
# print(model.country_model().predict(np.zeros(300).reshape(1, -1)))
