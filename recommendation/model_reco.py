import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

from db import DB
from definitions import ROOT_DIR
from models.tweet import Tweet
from recommendation.models.gmf_model import GMFModel
from recommendation.models.mf_model import MFModel
from recommendation.models.neumf_model import NeuMFModel
from recommendation.models.nnmf_model import NNMFModel


class ModelReco:

	def __init__(self, method, batch_size=256, num_negatives=4, num_factors_user=8, num_factors_item=8, regs=None,
				 nb_epochs=1, limit=None):
		assert method in ["gmf", "mf", "neumf", "nnmf"]

		if regs is None:
			regs = [0, 0]
		self.method = method
		self.batch_size = batch_size
		self.num_negatives = num_negatives
		self.num_factors_user = num_factors_user
		self.num_factors_item = num_factors_item
		self.nb_epochs = nb_epochs
		self.regs = regs
		self.train_corpus = None
		self.test_corpus = None
		self.num_users = None
		self.num_tweets = None
		self.model = None
		self.limit = limit

	def load_corpus(self):
		"""
		Load the corpus and the Favorite/RT adjancy matrix
		:param corpus_path: absolute path
		:param like_rt_graph: absolute path
		:return: pd.DataFrame object
		"""

		self.test_corpus = pd.read_csv(os.path.join(ROOT_DIR, 'corpus/model_reco/test_corpus.csv'))
		self.train_corpus = pd.read_csv(os.path.join(ROOT_DIR, 'corpus/model_reco/train_corpus.csv'))

		# self.test_corpus = self.test_corpus[self.test_corpus == 1]
		# self.train_corpus = self.train_corpus[self.train_corpus == 1]

		self.corpus = pd.concat([self.train_corpus, self.test_corpus], ignore_index=True)

		self.num_users = len(self.corpus.user_id.unique())
		self.num_tweets = len(self.corpus.id.unique())

		return self.train_corpus, self.test_corpus

	def create_model(self):
		"""
		Build and compile a MasterModel depending on the method asked
		:return:
		"""
		if self.method == "gmf":
			self.model = GMFModel(self.num_users, self.num_tweets, self.num_factors_user, self.num_factors_item,
								  self.regs).get_model()
		elif self.method == "mf":
			self.model = MFModel(self.num_users, self.num_tweets, self.num_factors_user, self.num_factors_item,
								 self.regs).get_model()
		elif self.method == "neumf":
			self.model = NeuMFModel(self.num_users, self.num_tweets, self.num_factors_user, self.num_factors_item,
									self.regs).get_model()
		elif self.method == "nnmf":
			self.model = NNMFModel(self.num_users, self.num_tweets, self.num_factors_user, self.num_factors_item,
								   self.regs).get_model()
		else:
			self.model = None
			raise Exception('Wgrong Argument ! : must be among "gmf", "mf", "neumf", "nnmf"')

		self.model.compile('adam', 'mean_squared_error')

	def train(self, save=False, display_metrics=False, train_all=False):
		"""
		Train the model
		:param save: boolean : save the model into a file
		:return:
		"""
		assert self.train_corpus is not None

		for e in range(self.nb_epochs):
			if train_all:
				self.model.fit([self.corpus.user_id, self.corpus.id],  # input
							   self.corpus.rating,  # labels
							   batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
			else:
				self.model.fit([self.train_corpus.user_id, self.train_corpus.id],  # input
							   self.train_corpus.rating,  # labels
							   batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)

			if save:
				self.save(epoch=e)
			if display_metrics:
				y_pred = self.predict()
				self.metrics(y_pred, epoch=e)

	def save(self, model_out_file=None, out_dir=os.path.join(ROOT_DIR, 'saved_models/reco_models'), epoch=0):
		"""
		Save the model into file
		:param model_out_file: name of the file
		:param out_dir: absolute path of the directory to save the model
		:param epoch: num of epoch
		:return: None
		"""
		if model_out_file is None:
			model_out_file = self.method

		if not os.path.exists(out_dir):
			os.mkdir(out_dir)

		# Saving weights
		self.model.save_weights(os.path.join(out_dir, model_out_file + str(epoch) + '.model'), overwrite=True)

		# Saving configuration
		f = open(os.path.join(out_dir, model_out_file + str(epoch)) + '.yaml', 'w')
		yaml_string = self.model.to_yaml()
		f.write(yaml_string)
		f.close()

	def get_reco(self, user, k_best=5):
		# from models.favorite import Favorite
		print(user.favs)
		# tweets that the user didn't fav
		negatives = [t[0] for t in
					 DB.get_instance().query(Tweet.id).filter(Tweet.id.notin_([f.tweet_id for f in user.favs])).all()]

		users = np.full(len(negatives), self.num_users + user.id, dtype='int32')
		predictions = self.model.predict([users, np.array(negatives)])

		predictions_map = []
		predictions = predictions.flatten()
		for i in range(len(predictions)):
			predictions_map.append({'tweet': negatives[i], 'predict': predictions[i]})

		print(predictions_map)

		predictions_map = sorted(predictions_map, key=lambda k: k['predict'], reverse=True)
		recommended_tweets = [Tweet.load(t['tweet']) for t in predictions_map[:k_best]]
		return recommended_tweets

	def predict(self):
		"""
		Predict values based on test corpus
		:return: predictions, 1-d array
		"""

		return self.model.predict([self.test_corpus.user_id, self.test_corpus.id])

	def metrics(self, predictions, epoch=None):
		"""
		Return the MAE metrics based on predictions
		:param predictions:
		:return:
		"""
		y_true = self.test_corpus.rating
		y_pred = np.round(predictions, 0)

		mae = mean_absolute_error(y_true, y_pred)
		acc = accuracy_score(y_true, y_pred)
		print('MAE', (('| epoch ' + str(epoch)) if epoch is not None else ''), self.method, mae)
		print('ACC', (('| epoch ' + str(epoch)) if epoch is not None else ''), self.method, acc)
		return mae, acc


if __name__ == '__main__':
	for method in ["gmf", "mf", "neumf", "nnmf"]:
		m = ModelReco(method)
		m.load_corpus()
		m.create_model()
		m.train()
		m.save()

		pred = m.predict()
		print('MAE ', method, m.metrics(pred))
