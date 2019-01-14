import numpy as np

from prediction_profile import PredictionProfile


class User:
	fname = 'users_profile.txt'

	def __init__(self, id=None, vec_size=300):
		next_id = User.next_id()

		if id is None:
			self.id = next_id
			self.vec = np.zeros(vec_size)
			self.nb_click = 0
			self.localisation = 'None'
			self.gender = 'None'
			self.emotion = 'None'
		else:
			if id >= next_id:
				raise Exception('This id does not exist.')
			self.id = id
			self.load()

	def update_profile(self, vec):

		self.nb_click += 1
		for i in range(len(self.vec)):
			self.vec[i] = (self.vec[i] * (self.nb_click - 1)) / self.nb_click + (vec[i] / self.nb_click)

		pp = PredictionProfile()
		self.localisation = pp.country_prediction(self.vec)
		self.gender = pp.gender_prediction(self.vec)
		self.emotion = pp.sentiment_prediction(self.vec)

	def save(self):
		users_data = {}  # user_id => line

		f = open(User.fname, "r")
		contents = f.readlines()
		i = 0
		for j in range(len(contents)):
			items = contents[j].split('\t')
			users_data[int(items[0])] = i
			i += 1
		f.close()

		to_insert = str(self.id) + '\t' + str(self.nb_click) + '\t' + str(
			list(self.vec)) + '\t' + self.localisation + '\t' + self.gender + '\t' + self.emotion + '\n'

		# if the id is not in the file
		if self.id not in users_data:
			contents.append(to_insert)
		else:
			contents[users_data[self.id]] = to_insert

		f = open(User.fname, "w")
		for l in contents:
			f.write(l)
		f.close()

	def load(self):
		assert self.id is not None

		f = open(User.fname, "r")
		for l in f:
			items = l.split('\t')
			if int(items[0]) == self.id:
				self.nb_click = int(items[1])
				self.vec = np.asarray([float(x) for x in items[2][1:-1].split(', ')])
				self.localisation = items[3]
				self.gender = items[4]
				self.emotion = items[5]
				f.close()
				return

	@staticmethod
	def next_id():
		f = open(User.fname, "r")
		contents = f.readlines()
		if len(contents) == 0:
			return 1
		return int(contents[-1].split('\t')[0]) + 1
