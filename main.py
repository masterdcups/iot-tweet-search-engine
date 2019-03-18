from recommendation.model_reco import ModelReco

if __name__ == '__main__':
	# create and update user example

	# BuildCorpus().call()
	# exit()

	for method in ["gmf", "mf", "neumf", "nnmf"]:
		m = ModelReco(method)
		train, test = m.load_corpus()
		m.create_model()
		m.train(train_all=True)
		from models.user import User

		u = User.load(user_name='clement')
		# u = None
		print(m.get_reco(u))

	exit()