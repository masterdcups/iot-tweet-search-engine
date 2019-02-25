from keras import Input, Model
from keras.layers import Embedding, Flatten, dot

from recommendation.models.master_model import MasterModel


class MFModel(MasterModel):

	def __init__(self, num_users, num_tweets, n_latent_factors_user=8, n_latent_factors_item=8, regs=None):
		if regs is None:
			regs = [0, 0]
		super().__init__(num_users, num_tweets, n_latent_factors_user, n_latent_factors_item, regs)

	def get_model(self):
		item_input = Input(shape=[1], name='Item')
		item_embedding = Embedding(self.num_tweets, self.n_latent_factors_item, name='Item-Embedding')(item_input)
		item_vec = Flatten(name='FlattenItem')(item_embedding)

		user_input = Input(shape=[1], name='User')
		user_vec = Flatten(name='FlattenUsers')(
			Embedding(self.num_users, self.n_latent_factors_user, name='User-Embedding')(user_input))

		# prod = merge([item_vec, user_vec], mode='dot', name='DotProduct')
		prod = dot([item_vec, user_vec], axes=1, normalize=False)
		return Model(input=[user_input, item_input], output=prod)
