from abc import abstractmethod


class MasterModel:

	def __init__(self, num_users, num_tweets, n_latent_factors_user=8, n_latent_factors_item=10, regs=[0, 0]):
		self.num_users = num_users
		self.num_tweets = num_tweets
		self.n_latent_factors_user = n_latent_factors_user
		self.n_latent_factors_item = n_latent_factors_item
		self.regs = regs

	@abstractmethod
	def get_model(self):
		return
