import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys
sys.path.append('../')
from keras import Input, Model
from keras.layers import Embedding, Flatten, concatenate, Dense, Dot, Dropout, BatchNormalization, dot
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.constraints import non_neg
from definitions import ROOT_DIR
from parser2 import Parser2


class tweetReco:

	def __init__(self, method):
		self.method = method

	def gmf_model(self, num_users, num_item, latent_dim, regs=[0, 0]):
		"""
		Method to generate the GMF model
		:param num_users: number of users
		:param num_item: number of tweets
		:param latent_dim: Embedding size
		:param regs: Regularization for user and item embeddings
		:return: GMF model
		"""
		user_input = Input(shape=(1,), dtype='int32', name='user_input')
		item_input = Input(shape=(1,), dtype='int32', name='item_input')

		MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='User-Embedding',
									  init='normal', W_regularizer=l2(regs[0]), input_length=1)
		MF_Embedding_Item = Embedding(input_dim=num_item, output_dim=latent_dim, name='Item-Embedding',
									  init='normal', W_regularizer=l2(regs[1]), input_length=1)

		# Crucial to flatten an embedding vector!
		user_latent = Flatten()(MF_Embedding_User(user_input))
		item_latent = Flatten()(MF_Embedding_Item(item_input))

		# Element-wise product of user and item embeddings
		predict_vector = concatenate([user_latent, item_latent])

		# Final prediction layer
		prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

		model = Model(input=[user_input, item_input], output=prediction)

		return model

	def neumf_model(self, n_users, n_item):
		"""
		Method to generate the NeuMF model
		:param n_users: number of users
		:param n_item: number of tweets
		:return: NeuMF model, model by default
		"""
		n_latent_factors_user = 8
		n_latent_factors_item = 10
		n_latent_factors_mf = 3

		item_input = Input(shape=[1], name='Item')
		item_embedding_mlp = Embedding(n_item, n_latent_factors_item, name='Item-Embedding-MLP')(
			item_input)
		item_vec_mlp = Flatten(name='FlattenItem-MLP')(item_embedding_mlp)
		item_vec_mlp = Dropout(0.2)(item_vec_mlp)

		item_embedding_mf = Embedding(n_item, n_latent_factors_mf, name='Item-Embedding')(
			item_input)
		item_vec_mf = Flatten(name='FlattenItem-MF')(item_embedding_mf)
		item_vec_mf = Dropout(0.2)(item_vec_mf)

		user_input = Input(shape=[1], name='User')
		user_vec_mlp = Flatten(name='FlattenUsers-MLP')(
			Embedding(n_users, n_latent_factors_user, name='User-Embedding-MLP')(user_input))
		user_vec_mlp = Dropout(0.2)(user_vec_mlp)

		user_vec_mf = Flatten(name='FlattenUsers-MF')(
			Embedding(n_users, n_latent_factors_mf, name='User-Embedding')(user_input))
		user_vec_mf = Dropout(0.2)(user_vec_mf)

		concat = concatenate([item_vec_mlp, user_vec_mlp])
		concat_dropout = Dropout(0.2)(concat)
		dense = Dense(200, name='FullyConnected')(concat_dropout)
		dense_batch = BatchNormalization(name='Batch')(dense)
		dropout_1 = Dropout(0.2, name='Dropout-1')(dense_batch)
		dense_2 = Dense(100, name='FullyConnected-1')(dropout_1)
		dense_batch_2 = BatchNormalization(name='Batch-2')(dense_2)

		dropout_2 = Dropout(0.2, name='Dropout-2')(dense_batch_2)
		dense_3 = Dense(50, name='FullyConnected-2')(dropout_2)
		dense_4 = Dense(20, name='FullyConnected-3', activation='relu')(dense_3)

		pred_mf = concatenate([item_vec_mf, user_vec_mf])

		pred_mlp = Dense(1, activation='relu', name='Activation')(dense_4)

		combine_mlp_mf = concatenate([pred_mf, pred_mlp])
		result_combine = Dense(100, name='Combine-MF-MLP')(combine_mlp_mf)
		deep_combine = Dense(100, name='FullyConnected-4')(result_combine)

		result = Dense(1, name='Prediction')(deep_combine)

		model = Model(input=[user_input, item_input], output=result)

		return model

	def mf_model(self, n_users, n_item):
		"""
		Method to generate the MF model
		:param n_users: number of users
		:param n_item: number of tweets
		:return: MF model
		"""
		n_latent_factors = 8

		item_input = Input(shape=[1], name='Item')
		item_embedding = Embedding(n_item, n_latent_factors, name='Item-Embedding')(item_input)
		item_vec = Flatten(name='FlattenItem')(item_embedding)

		user_input = Input(shape=[1], name='User')
		user_vec = Flatten(name='FlattenUsers')(
			Embedding(n_users, n_latent_factors, name='User-Embedding')(user_input))

		# prod = merge([item_vec, user_vec], mode='dot', name='DotProduct')
		prod = dot([item_vec, user_vec], axes=1, normalize=False)
		model = Model(input=[user_input, item_input], output=prod)
		# model.compile('adam', 'mean_squared_error')

		return model

	def nnmf_model(self, n_users, n_item):
		"""
		Method to generate the NNMF model
		:param n_users: number of users
		:param n_item: number of tweets
		:return: NNMF model
		"""
		n_latent_factors = 8

		item_input = Input(shape=[1], name='Item')
		item_embedding = Embedding(n_item, n_latent_factors, name='Item-Embedding',
								   embeddings_constraint=non_neg())(item_input)
		item_vec = Flatten(name='FlattenItem')(item_embedding)

		user_input = Input(shape=[1], name='User')
		user_vec = Flatten(name='FlattenUsers')(
			Embedding(n_users, n_latent_factors, name='User-Embedding',
					  embeddings_constraint=non_neg())(user_input))

		prod = dot([item_vec, user_vec], axes=1, name='DotProduct')
		model = Model(input=[user_input, item_input], output=prod)

		return model

	def get_train_instances(self, train, num_negatives, num_item):
		"""
		Method to train the instances
		:param train: part of the corpus
		:param num_negatives: number of negative instances to pair with a positive instance
		:param num_item: number of tweet
		:return: user_input, item_input, labels
		"""
		user_input, item_input, labels = [], [], []

		for index, line in train.iterrows():

			u = line.User_ID_u
			i = line.TweetID_u

			# positive instance
			user_input.append(u)
			item_input.append(i)
			labels.append(1)

			# negative instances
			for t in range(num_negatives):
				j = np.random.randint(num_item)
				while (u, j) in train.keys():
					j = np.random.randint(num_item)
				user_input.append(u)
				item_input.append(j)
				labels.append(0)
		return user_input, item_input, labels

	def get_negatives_tweets(self, matrix, user_id):
		"""
		Method to get the negatives tweets
		:param matrix: user item matrix
		:param user_id: main user
		:return: tweets
		"""
		tweets = []
		num_users, num_tweets = matrix.shape

		# negative instances
		for t in range(num_tweets):
			if (user_id, t) not in matrix.keys():
				tweets.append(t)

		return tweets

	def build_matrix(self):
		"""
		Method to build the user item matrix
		:return: matrix
		"""
		corpus = Parser2.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v31.tsv'),
												   categorize=True)

		num_users = corpus.User_ID_u.max() + 1
		num_tweets = corpus.TweetID_u.max() + 1

		# matrix construction
		mat = sp.dok_matrix((num_users, num_tweets), dtype=np.float32)

		for index, tweet in corpus.iterrows():
			mat[int(tweet.User_ID_u), int(tweet.TweetID_u)] = 1.

		like_rt_file = open(os.path.join(ROOT_DIR, 'corpus/like_rt_graph.adj'), 'r')
		for line in like_rt_file:
			parts = line[:-1].split(' ')
			user_name = parts[0]
			user_id = corpus[corpus.User_Name == user_name].User_ID_u.iloc[0]
			tweets = parts[1:]

			for tweet in tweets:
				if len(corpus[corpus.TweetID == int(tweet)]) > 0:
					tweet_id = corpus[corpus.TweetID == int(tweet)].TweetID_u.iloc[0]
					mat[user_id, tweet_id] = 1.

		like_rt_file.close()

		return mat

	def save_model(self, model_out_file, model, epoch):
		"""
		Method to save the generated model
		:param model_out_file:
		:param model: generated model
		:param epoch: 1
		"""
		model.save_weights(model_out_file + str(epoch) + '.model', overwrite=True)
		f = open(model_out_file + str(epoch) + '.yaml', 'w')
		yaml_string = model.to_yaml()
		f.write(yaml_string)
		f.close()

	def load_corpus(self, num_negatives):
		"""
		Method to load the corpus
		:param num_negatives: number of negatives tweets
		:return: num_users, num_tweets, train, test
		"""
		original_corpus = Parser2.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v31.tsv'),
															categorize=True)
		num_users = original_corpus.User_ID_u.max() + 1
		num_tweets = original_corpus.TweetID_u.max() + 1

		corpus = original_corpus[['User_ID_u', 'TweetID_u']]

		like_rt_file = open(os.path.join(ROOT_DIR, 'corpus/like_rt_graph.adj'), 'r')
		for line in like_rt_file:
			parts = line[:-1].split(' ')
			user_name = parts[0]
			user_id = original_corpus[original_corpus.User_Name == user_name].User_ID_u.iloc[0]
			tweets = parts[1:]

			for tweet in tweets:
				if len(original_corpus[original_corpus.TweetID == int(tweet)]) > 0:
					tweet_id = original_corpus[original_corpus.TweetID == int(tweet)].TweetID_u.iloc[0]

					corpus = corpus.append({'User_ID_u': user_id, 'TweetID_u': tweet_id}, ignore_index=True)

		like_rt_file.close()

		corpus['Rating'] = 1

		for index, line in corpus.iterrows():
			u = line.User_ID_u

			# negative instances
			for t in range(num_negatives):
				j = np.random.randint(num_tweets)
				while (u, j) in corpus[['User_ID_u', 'TweetID_u']]:
					j = np.random.randint(num_tweets)

				corpus = corpus.append({'User_ID_u': u, 'TweetID_u': j, 'Rating': 0}, ignore_index=True)

		train, test = train_test_split(corpus, test_size=0.2)

		return num_users, num_tweets, train, test

	def model_creation(self):
		"""
		Method to create a model
		:return: model
		"""
		num_negatives = 4  # Number of negative instances to pair with a positive instance.
		batch_size = 256
		regs = [0, 0]  # Regularization for user and item embeddings.
		num_factors = 8  # Embedding size

		num_users, num_tweets, train, test = self.load_corpus(num_negatives)

		# Build model
		if self.method == "gmf":
			model = self.gmf_model(num_users, num_tweets, num_factors, regs)
		elif self.method == "mf":
			model = self.mf_model(num_users, num_tweets)
		elif self.method == "nnmf":
			model = self.nnmf_model(num_users, num_tweets)
		else:
			model = self.neumf_model(num_users, num_tweets)

		# ~~ Compiling model
		# model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
		model.compile('adam', 'mean_squared_error')

		# Training
		model.fit([train.User_ID_u, train.TweetID_u],  # input
						 train.Rating,  # labels
						 batch_size=batch_size, epochs=10, verbose=0, shuffle=True)

		model.predict([test.User_ID_u, test.TweetID_u])

		item_embedding_learnt = model.get_layer(name='Item-Embedding').get_weights()[0]
		pd.DataFrame(item_embedding_learnt).describe()

		user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
		pd.DataFrame(user_embedding_learnt).describe()

		return model


if __name__ == '__main__':

	# method : gmf, neumf, mf or nnmf
	# by default neumf
	reco = tweetReco("gmf")
	model = reco.model_creation()
	main_user = 0

	# 1. getting the negatives tweets
	mat = reco.build_matrix()
	negatives = reco.get_negatives_tweets(mat, main_user)
	# print(negatives)

	# prediction
	users = np.full(len(negatives), main_user, dtype='int32')
	predictions = model.predict([users, np.array(negatives)], batch_size=100, verbose=0)

	predictions_map = []
	predictions = predictions.flatten()
	for i in range(len(predictions)):
		predictions_map.append({'tweet': negatives[i], 'predict': predictions[i]})

	predictions_map = sorted(predictions_map, key=lambda k: k['predict'], reverse=True)
	k_first = 5
	recommended_tweets = [t['tweet'] for t in predictions_map[:k_first]]
	print(recommended_tweets)
