import numpy as np
from keras.layers import Embedding, Input, Dense, Flatten, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

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


def get_train_instances(train, num_negatives, num_tweets):
	user_input, item_input, labels = [], [], []

	for index, tweet in train.iterrows():
		u = tweet.User_ID
		i = tweet.TweetID
		# positive instance
		user_input.append(u)
		item_input.append(i)
		labels.append(1)
		# negative instances
		for t in range(num_negatives):
			j = np.random.randint(num_tweets)
			while (u, j) in train[['User_ID', 'TweetID']]:
				j = np.random.randint(num_tweets)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	return user_input, item_input, labels


class GMFModel:

	def __init__(self):
		return


if __name__ == '__main__':
	corpus = Parser.parsing_iot_corpus_pandas('../corpus/iot-tweets-vector-new.tsv')

	num_negatives = 4  # Number of negative instances to pair with a positive instance.
	regs = [0, 0]  # Regularization for user and item embeddings.
	num_factors = 8  # Embedding size
	epochs = 1  # Number of epochs
	batch_size = 256
	learning_rate = 0.001
	model_out_file = 'train_'

	num_users = corpus.User_ID.max() + 1
	num_tweets = corpus.TweetID.max() + 1

	print(num_users, 'users')
	print(num_tweets, 'tweets')

	# corpus = corpus[corpus.User_ID >= 0]
	train, test = train_test_split(corpus, test_size=0.2)

	# Build model
	model = get_model(num_users, num_tweets, num_factors, regs)

	# Compiling model
	model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

	for epoch in range(epochs):
		# Generate training instances
		user_input, item_input, labels = get_train_instances(train, num_negatives, num_tweets)
		print(user_input)
		print(item_input)
		print(labels)

		# Training
		hist = model.fit([np.array(user_input), np.array(item_input)],  # input
						 np.array(labels),  # labels
						 batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

		model.save_weights(model_out_file + str(epoch) + '.model', overwrite=True)
