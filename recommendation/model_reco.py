import os

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from definitions import ROOT_DIR
from parser import Parser
from recommendation.models.gmf_model import GMFModel
from recommendation.models.mf_model import MFModel
from recommendation.models.neumf_model import NeuMFModel
from recommendation.models.nnmf_model import NNMFModel


class ModelReco:

	def __init__(self, method, batch_size=256, num_negatives=4, num_factors_user=8, num_factors_item=8, regs=None,
				 nb_epochs=1):
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

	def load_corpus(self, corpus_path=os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v31.tsv'),
					like_rt_graph=os.path.join(ROOT_DIR, 'corpus/like_rt_graph.adj')):
		"""
		Load the corpus and the Favorite/RT adjancy matrix
		:param corpus_path: absolute path
		:param like_rt_graph: absolute path
		:return: pd.DataFrame object
		"""

		original_corpus = Parser.parsing_base_corpus_pandas(corpus_path, categorize=True)

		self.num_users = len(original_corpus.User_Name.unique())
		self.num_tweets = len(original_corpus.TweetID.unique())

		corpus = original_corpus[['User_ID_u', 'TweetID_u']]

		like_rt_file = open(like_rt_graph, 'r')
		for line in like_rt_file:
			parts = line[:-1].split(' ')
			user_name = parts[0]
			user_id = original_corpus[original_corpus.User_Name == user_name].User_ID_u.iloc[0]
			tweets = parts[1:]  # like or RT tweets

			for tweet in tweets:
				if len(original_corpus[original_corpus.TweetID == int(tweet)]) > 0:
					tweet_id = original_corpus[original_corpus.TweetID == int(tweet)].TweetID_u.iloc[0]

					corpus = corpus.append({'User_ID_u': user_id, 'TweetID_u': tweet_id}, ignore_index=True)

		like_rt_file.close()

		corpus['Rating'] = 1

		for index, line in corpus.iterrows():
			u = line.User_ID_u

			# negative instances
			for t in range(self.num_negatives):
				j = np.random.randint(self.num_tweets)
				while (u, j) in corpus[['User_ID_u', 'TweetID_u']]:
					j = np.random.randint(self.num_tweets)

				corpus = corpus.append({'User_ID_u': u, 'TweetID_u': j, 'Rating': 0}, ignore_index=True)

		self.train_corpus, self.test_corpus = train_test_split(corpus, test_size=0.2)
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

	def train(self, save=False):
		"""
		Train the model
		:param save: boolean : save the model into a file
		:return:
		"""
		assert self.train_corpus is not None

		for e in range(self.nb_epochs):
			self.model.fit([self.train_corpus.User_ID_u, self.train_corpus.TweetID_u],  # input
						   self.train_corpus.Rating,  # labels
						   batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)

			if save:
				m.save(epoch=e)

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

	def predict(self):
		"""
		Predict values based on test corpus
		:return: predictions, 1-d array
		"""
		return self.model.predict([self.test_corpus.User_ID_u, self.test_corpus.TweetID_u])

	def mae_metric(self, predictions):
		"""
		Return the MAE metrics based on predictions
		:param predictions:
		:return:
		"""
		y_true = self.test_corpus.Rating
		y_hat = np.round(predictions, 0)

		mae = mean_absolute_error(y_true, y_hat)
		return mae


if __name__ == '__main__':
	for method in ["gmf", "mf", "neumf", "nnmf"]:
		m = ModelReco(method)
		m.load_corpus(corpus_path=os.path.join(ROOT_DIR, 'corpus/iot-tweets-2009-2016-completv3.tsv'),
					  like_rt_graph=os.path.join(ROOT_DIR, 'corpus/likes_matrix.tsv'))
		m.create_model()
		m.train()
		m.save()

		pred = m.predict()
		print('MAE ', method, m.mae_metric(pred))

# users = np.full(len(negatives), main_user, dtype='int32')
# predictions = model.predict([users, np.array(negatives)], batch_size=100, verbose=0)
#
# predictions_map = []
# predictions = predictions.flatten()
# for i in range(len(predictions)):
# 	predictions_map.append({'tweet': negatives[i], 'predict': predictions[i]})
#
# predictions_map = sorted(predictions_map, key=lambda k: k['predict'], reverse=True)
# k_first = 5
# recommended_tweets = [t['tweet'] for t in predictions_map[:k_first]]
# print(recommended_tweets)
