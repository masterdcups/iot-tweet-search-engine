from keras import Input, Model
from keras.layers import Embedding, Flatten, concatenate, Dense, Dropout, BatchNormalization

from recommendation.models.master_model import MasterModel


class NeuMFModel(MasterModel):

	def __init__(self, num_users, num_tweets, n_latent_factors_user=8, n_latent_factors_item=10, regs=None):
		if regs is None:
			regs = [0, 0]

		super().__init__(num_users, num_tweets, n_latent_factors_user, n_latent_factors_item, regs)

	def get_model(self):
		n_latent_factors_mf = 3

		item_input = Input(shape=[1], name='Item')
		item_embedding_mlp = Embedding(self.num_tweets, self.n_latent_factors_item, name='Item-Embedding-MLP')(
			item_input)
		item_vec_mlp = Flatten(name='FlattenItem-MLP')(item_embedding_mlp)
		item_vec_mlp = Dropout(0.2)(item_vec_mlp)

		item_embedding_mf = Embedding(self.num_tweets, n_latent_factors_mf, name='Item-Embedding')(
			item_input)
		item_vec_mf = Flatten(name='FlattenItem-MF')(item_embedding_mf)
		item_vec_mf = Dropout(0.2)(item_vec_mf)

		user_input = Input(shape=[1], name='User')
		user_vec_mlp = Flatten(name='FlattenUsers-MLP')(
			Embedding(self.num_users, self.n_latent_factors_user, name='User-Embedding-MLP')(user_input))
		user_vec_mlp = Dropout(0.2)(user_vec_mlp)

		user_vec_mf = Flatten(name='FlattenUsers-MF')(
			Embedding(self.num_users, n_latent_factors_mf, name='User-Embedding')(user_input))
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

		return Model(input=[user_input, item_input], output=result)
