from keras import Input, Model
from keras.layers import Embedding, Flatten, concatenate, Dense
from keras.regularizers import l2

from recommendation.models.master_model import MasterModel


class GMFModel(MasterModel):

	def __init__(self, num_users, num_tweets, n_latent_factors_user=8, n_latent_factors_item=8, regs=None):
		if regs is None:
			regs = [0, 0]
		super().__init__(num_users, num_tweets, n_latent_factors_user, n_latent_factors_item, regs)

	def get_model(self):
		user_input = Input(shape=(1,), dtype='int32', name='user_input')
		item_input = Input(shape=(1,), dtype='int32', name='item_input')

		mf_embedding_user = Embedding(input_dim=self.num_users, output_dim=self.n_latent_factors_user,
									  name='User-Embedding',
									  init='normal', W_regularizer=l2(self.regs[0]), input_length=1)
		mf_embedding_item = Embedding(input_dim=self.num_tweets, output_dim=self.n_latent_factors_item,
									  name='Item-Embedding',
									  init='normal', W_regularizer=l2(self.regs[1]), input_length=1)

		# Crucial to flatten an embedding vector!
		user_latent = Flatten()(mf_embedding_user(user_input))
		item_latent = Flatten()(mf_embedding_item(item_input))

		# Element-wise product of user and item embeddings
		predict_vector = concatenate([user_latent, item_latent])

		# Final prediction layer
		prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

		return Model(input=[user_input, item_input], output=prediction)
