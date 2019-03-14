import numpy as np
from sklearn.preprocessing import OneHotEncoder

from db import DB
from models.tweet import Tweet


class ProfileOneHotEncoder:
	enc = None

	@staticmethod
	def get_instance():
		if ProfileOneHotEncoder.enc is None:
			ProfileOneHotEncoder.enc = ProfileOneHotEncoder._load_enc()
		return ProfileOneHotEncoder.enc

	@staticmethod
	def _load_enc():
		encoder = OneHotEncoder()
		X = []
		query = DB.get_instance().query(Tweet.gender, Tweet.country, Tweet.sentiment).distinct(Tweet.country).group_by(
			Tweet.country, Tweet.gender, Tweet.sentiment)
		for gender, country, sentiment in query.all():
			X.append([gender, country, sentiment])
		encoder.fit(X)
		return encoder

	@staticmethod
	def add_info_to_vec(vector, gender, location, sentiment):
		infos_vec = ProfileOneHotEncoder.get_instance().transform([[gender, location, sentiment]]).toarray()[0]
		return np.concatenate((vector, np.array(infos_vec)))


if __name__ == '__main__':
	print(ProfileOneHotEncoder.add_info_to_vec(np.zeros(300), 'male', 'rw', 'neutral'))
