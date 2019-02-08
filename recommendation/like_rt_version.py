import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from keras import Input, Model
from keras.layers import Embedding, Flatten, concatenate, Dense, Dot, Dropout, BatchNormalization, dot
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from definitions import ROOT_DIR
from parser import Parser


def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
	user_input = Input(shape=(1,), dtype='int32', name='user_input')
	item_input = Input(shape=(1,), dtype='int32', name='item_input')

	MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
								  init='normal', W_regularizer=l2(regs[0]), input_length=1)
	MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
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


def neumf_model(n_users, n_movies):
	n_latent_factors_user = 8
	n_latent_factors_movie = 10
	n_latent_factors_mf = 3

	movie_input = Input(shape=[1], name='Item')
	movie_embedding_mlp = Embedding(n_movies, n_latent_factors_movie, name='Movie-Embedding-MLP')(
		movie_input)
	movie_vec_mlp = Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
	movie_vec_mlp = Dropout(0.2)(movie_vec_mlp)

	movie_embedding_mf = Embedding(n_movies, n_latent_factors_mf, name='Movie-Embedding-MF')(
		movie_input)
	movie_vec_mf = Flatten(name='FlattenMovies-MF')(movie_embedding_mf)
	movie_vec_mf = Dropout(0.2)(movie_vec_mf)

	user_input = Input(shape=[1], name='User')
	user_vec_mlp = Flatten(name='FlattenUsers-MLP')(
		Embedding(n_users, n_latent_factors_user, name='User-Embedding-MLP')(user_input))
	user_vec_mlp = Dropout(0.2)(user_vec_mlp)

	user_vec_mf = Flatten(name='FlattenUsers-MF')(
		Embedding(n_users, n_latent_factors_mf, name='User-Embedding-MF')(user_input))
	user_vec_mf = Dropout(0.2)(user_vec_mf)

	concat = concatenate([movie_vec_mlp, user_vec_mlp])
	concat_dropout = Dropout(0.2)(concat)
	dense = Dense(200, name='FullyConnected')(concat_dropout)
	dense_batch = BatchNormalization(name='Batch')(dense)
	dropout_1 = Dropout(0.2, name='Dropout-1')(dense_batch)
	dense_2 = Dense(100, name='FullyConnected-1')(dropout_1)
	dense_batch_2 = BatchNormalization(name='Batch-2')(dense_2)

	dropout_2 = Dropout(0.2, name='Dropout-2')(dense_batch_2)
	dense_3 = Dense(50, name='FullyConnected-2')(dropout_2)
	dense_4 = Dense(20, name='FullyConnected-3', activation='relu')(dense_3)

	pred_mf = Dot([movie_vec_mf, user_vec_mf])
	# pred_mf = concatenate([movie_vec_mf, user_vec_mf])

	pred_mlp = Dense(1, activation='relu', name='Activation')(dense_4)

	combine_mlp_mf = concatenate([pred_mf, pred_mlp])
	result_combine = Dense(100, name='Combine-MF-MLP')(combine_mlp_mf)
	deep_combine = Dense(100, name='FullyConnected-4')(result_combine)

	result = Dense(1, name='Prediction')(deep_combine)

	model = Model(input=[user_input, movie_input], output=result)

	return model


def mf_model(n_users, n_movies):
	n_latent_factors = 8

	movie_input = Input(shape=[1], name='Item')
	movie_embedding = Embedding(n_movies, n_latent_factors, name='Movie-Embedding')(movie_input)
	movie_vec = Flatten(name='FlattenMovies')(movie_embedding)

	user_input = Input(shape=[1], name='User')
	user_vec = Flatten(name='FlattenUsers')(
		Embedding(n_users, n_latent_factors, name='User-Embedding')(user_input))

	# prod = merge([movie_vec, user_vec], mode='dot', name='DotProduct')
	prod = dot([movie_vec, user_vec], axes=1, normalize=False)
	model = Model(input=[user_input, movie_input], output=prod)
	# model.compile('adam', 'mean_squared_error')

	return model


def nnmf_model(n_users, n_movies):
	n_latent_factors = 8

	from keras.constraints import non_neg

	movie_input = Input(shape=[1], name='Item')
	movie_embedding = Embedding(n_movies, n_latent_factors, name='NonNegMovie-Embedding',
								embeddings_constraint=non_neg())(movie_input)
	movie_vec = Flatten(name='FlattenMovies')(movie_embedding)

	user_input = Input(shape=[1], name='User')
	user_vec = Flatten(name='FlattenUsers')(
		Embedding(n_users, n_latent_factors, name='NonNegUser-Embedding',
				  embeddings_constraint=non_neg())(user_input))

	prod = dot([movie_vec, user_vec], axes=1, name='DotProduct')
	model = Model(input=[user_input, movie_input], output=prod)

	return model


def get_train_instances(train, num_negatives, num_items):
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
			j = np.random.randint(num_items)
			while (u, j) in train.keys():
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	return user_input, item_input, labels


def get_negatives_tweets(matrix, user_id):
	tweets = []
	num_users, num_tweets = matrix.shape

	# negative instances
	for t in range(num_tweets):
		if (user_id, t) not in matrix.keys():
			tweets.append(t)

	return tweets


def build_matrix():
	corpus = Parser.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-10.tsv'),
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


def save_model(model_out_file, model, epoch):
	model.save_weights(model_out_file + str(epoch) + '.model', overwrite=True)
	f = open(model_out_file + str(epoch) + '.yaml', 'w')
	yaml_string = model.to_yaml()
	f.write(yaml_string)
	f.close()


def load_corpus(num_negatives):
	original_corpus = Parser.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-10.tsv'),
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


if __name__ == '__main__':
	num_negatives = 4  # Number of negative instances to pair with a positive instance.
	regs = [0, 0]  # Regularization for user and item embeddings.
	num_factors = 8  # Embedding size
	epochs = 1  # Number of epochs
	batch_size = 256
	learning_rate = 0.001
	model_out_file = 'train_'

	num_users, num_tweets, train, test = load_corpus(num_negatives)

	# Build model
	# model = get_model(num_users, num_tweets, num_factors, regs)
	# model = neumf_model(num_users, num_tweets)
	model = mf_model(num_users, num_tweets)

	# Compiling model
	# model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
	model.compile('adam', 'mean_squared_error')

	y_true = test.Rating

	# for epoch in range(epochs):
	# Generate training instances

	# user_input = train.User_ID_u
	# item_input = train.TweetID_u
	# labels = train.Rating

	# Training
	hist = model.fit([train.User_ID_u, train.TweetID_u],  # input
					 train.Rating,  # labels
					 batch_size=batch_size, epochs=10, verbose=0, shuffle=True)

	# save model in file
	# save_model(model_out_file, model, 0)

	map_true_pred_comparaison = pd.DataFrame(columns=['True_', 'Pred'])

	from sklearn.metrics import mean_absolute_error

	predictions = model.predict([test.User_ID_u, test.TweetID_u])

	print(predictions)
	print(y_true)
	# print(np.array(y_true))
	#
	# map_true_pred_comparaison.Pred = predictions.flatten()
	# map_true_pred_comparaison.True_ = np.array(y_true)
	#
	# map_true_pred_comparaison = map_true_pred_comparaison.sort_values(by=['Pred'])
	#
	# print(map_true_pred_comparaison)

	y_hat = np.round(predictions, 0)
	print(y_hat)
	print(mean_absolute_error(y_true, y_hat))
	print(mean_absolute_error(y_true, model.predict([test.User_ID_u, test.TweetID_u])))

	movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
	print(pd.DataFrame(movie_embedding_learnt).describe())

	user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
	print(pd.DataFrame(user_embedding_learnt).describe())

	exit()

	main_user = 0

	# 1. getting the negatives tweets
	negatives = get_negatives_tweets(mat, main_user)
	print(negatives)

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
