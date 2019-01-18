import numpy as np

from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier


class User:
	user_fname = 'users_profile.tsv'
	author_fname = 'authors_profile.tsv'

	def __init__(self, id=None, nb_click=None, vector=None, localisation=None, gender=None, emotion=None,
				 topic_vector=None, centrality=None, vec_size=300):
		next_id = User.next_id()

		if id is None:
			self.id = next_id
			self.vec = np.zeros(vec_size)
			self.nb_click = 0
			self.localisation = None
			self.gender = None
			self.emotion = None
			self.topic_vector = None
			self.centrality = None
		else:
			self.id = id
			self.vec = vector
			self.nb_click = nb_click
			self.localisation = localisation
			self.gender = gender
			self.emotion = emotion
			self.topic_vector = topic_vector
			self.centrality = centrality

	def update_profile(self, vec):

		self.nb_click += 1
		for i in range(len(self.vec)):
			self.vec[i] = (self.vec[i] * (self.nb_click - 1)) / self.nb_click + (vec[i] / self.nb_click)

		pp = PredictionProfile()
		tcf = TopicsClassifier()
		self.localisation = pp.country_prediction(self.vec)
		self.gender = pp.gender_prediction(self.vec)
		self.emotion = pp.sentiment_prediction(self.vec)
		self.topic_vector = tcf.predict(vec.reshape(1, -1))[0]

	def save(self):
		users_data = {}  # user_id => line

		f = open(User.user_fname if type(self.id) is int else User.author_fname, "r")
		contents = f.readlines()
		i = 0
		for j in range(len(contents)):
			items = contents[j].split('\t')
			users_data[int(items[0])] = i
			i += 1
		f.close()

		to_insert = str(self.id) + '\t' + str(self.nb_click) + '\t' + str(
			list(self.vec)) + '\t' + self.localisation + '\t' + self.gender + '\t' + self.emotion + '\t' + str(
			list(self.topic_vector)) + ('\t' + str(self.centrality)) if type(self.id) is int else '' + '\n'

		# if the id is not in the file
		if self.id not in users_data:
			contents.append(to_insert)
		else:
			contents[users_data[self.id]] = to_insert

		f = open(User.user_fname, "w")
		for l in contents:
			f.write(l)
		f.close()

	def load(self):
		assert self.id is not None

		f = open(User.user_fname if type(self.id) is int else User.author_fname, "r")
		for l in f:
			items = l.split('\t')
			if items[0] == str(self.id):
				self.nb_click = int(items[1])
				self.vec = np.asarray([float(x) for x in items[2][1:-1].split(', ')])
				self.localisation = items[3]
				self.gender = items[4]
				self.emotion = items[5]
				self.topic_vector = np.asarray([float(x) for x in items[6][1:-2].split(', ')])
				if type(id) is str:
					self.centrality = float(items[7])
				f.close()
				return

	@staticmethod
	def next_id():
		f = open(User.user_fname, "r")
		contents = f.readlines()
		if len(contents) == 0:
			return 1
		return int(contents[-1].split('\t')[0]) + 1

	@staticmethod
	def get_all_authors():
		users = []
		file = open(User.author_fname, "r")
		for line in file:
			items = line.split('\t')
			u = User(
				id=items[0],
				nb_click=int(items[1]),
				vector=np.asarray([float(x) for x in items[2][1:-1].split(', ')]),
				localisation=items[3],
				gender=items[4],
				emotion=items[5],
				topic_vector=np.asarray([float(x) for x in items[6][1:-2].split(', ')]),
				centrality=float(items[7])
			)
			users.append(u)
		file.close()
		return users

# def load_graph(self, filename=user_graph_path):
# 	if self.graph is None:
# 		self.graph = nx.DiGraph(nx.read_adjlist(filename))
