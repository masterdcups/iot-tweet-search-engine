import os

import networkx as nx
import numpy as np

from definitions import ROOT_DIR
from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier


class User:
	user_fname = os.path.join(ROOT_DIR, 'saved_models/users_profile.tsv')
	author_fname = os.path.join(ROOT_DIR, 'saved_models/authors_profile.tsv')
	user_graph_path = os.path.join(ROOT_DIR, 'corpus/author_graph.net')

	def __init__(self, id=None, nb_click=0, vector=np.zeros(300), localisation='', gender='', emotion='',
				 topic_vector=np.asarray([]), centrality=0., vec_size=300):

		if id is None:
			self.id = User.next_id()
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

	def predict_profile(self):
		"""
		Call all the predictions models to fill the localisation, gender, etc
		:return:
		"""
		pp = PredictionProfile()
		tcf = TopicsClassifier()
		self.localisation = pp.country_prediction(self.vec)
		self.gender = pp.gender_prediction(self.vec)
		self.emotion = pp.sentiment_prediction(self.vec)
		self.topic_vector = tcf.predict(self.vec.reshape(1, -1))[0]

	def update_profile(self, vec, predict=True):
		"""
		Update the profile of the user with the new vec param
		:param vec: (np.array) vector of the tweet to add
		:param predict: (boolean) whether to predict localisation, gender, etc or not
		:return:
		"""
		self.nb_click += 1
		for i in range(len(self.vec)):
			self.vec[i] = (self.vec[i] * (self.nb_click - 1)) / self.nb_click + (vec[i] / self.nb_click)

		if predict:
			self.predict_profile()

	def save(self):
		"""
		Save the user in the corresponding file
		:return:
		"""
		users_data = {}  # user_id => line

		self.create_files()
		f = open(User.user_fname if type(self.id) is int else User.author_fname, "r")
		contents = f.readlines()
		i = 0
		for j in range(len(contents)):
			items = contents[j].split('\t')
			id = int(items[0]) if type(self.id) is int else items[0]
			users_data[id] = i
			i += 1
		f.close()

		to_insert = str(self.id) + '\t' + str(self.nb_click) + '\t' + str(
			list(self.vec)) + '\t' + self.localisation + '\t' + self.gender + '\t' + self.emotion + '\t' + str(
			list(self.topic_vector)) + (('\t' + str(self.centrality)) if type(self.id) is str else '') + '\n'

		# if the id is not in the file
		if self.id not in users_data:
			contents.append(to_insert)
		else:
			contents[users_data[self.id]] = to_insert

		f = open(User.user_fname if type(self.id) is int else User.author_fname, "w")
		for l in contents:
			f.write(l)
		f.close()

	def load(self):
		"""Load the user from the coresponding file of do nothing"""
		assert self.id is not None

		self.create_files()
		f = open(User.user_fname if type(self.id) is int else User.author_fname, "r")
		for l in f:
			items = l.split('\t')
			if items[0] == str(self.id):
				self.nb_click = int(items[1])
				self.vec = np.asarray([float(x) for x in items[2][1:-1].split(', ')])
				self.localisation = items[3]
				self.gender = items[4]
				self.emotion = items[5]
				self.topic_vector = np.asarray([]) if items[6] == '[]' else np.asarray(
					[float(x) for x in items[6][1:-1].split(', ')])
				if type(id) is str:
					self.centrality = float(items[7])
				f.close()
				return

	@staticmethod
	def next_id():
		"""Get the max +1 id in the file"""
		f = open(User.user_fname, "r")
		contents = f.readlines()
		if len(contents) == 0:
			return 1
		return int(contents[-1].split('\t')[0]) + 1

	@staticmethod
	def get_all_authors():
		"""
		Fetch all the authors from the tweets
		:return:
		"""
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
				topic_vector=np.asarray([]) if items[6] == '[]' else np.asarray(
					[float(x) for x in items[6][1:-1].split(', ')]),
				centrality=float(items[7])
			)
			users.append(u)
		file.close()
		return users

	def create_files(self):
		"""
		Create the users and authors files if they don't exists
		:return:
		"""
		if type(self.id) is int and not os.path.exists(User.user_fname):
			open(User.user_fname, 'w+')

		if type(self.id) is str and not os.path.exists(User.author_fname):
			open(User.author_fname, 'w+')

	@staticmethod
	def create_authors(corpus):
		"""
		Generate the authors_profile.tsv file
		To perform just ONE time
		:return:
		"""
		for tweet in corpus:
			u = User(tweet['Author'])
			u.load()
			u.update_profile(tweet['Vector'], predict=False)
			u.save()

		graph = User.load_graph()
		centralities = nx.eigenvector_centrality(graph)
		for author in User.get_all_authors():
			author.centrality = centralities[author.id] if author.id in centralities else 0.
			author.predict_profile()
			author.save()
		return

	@staticmethod
	def load_graph(filename=user_graph_path):
		return nx.DiGraph(nx.read_adjlist(filename))


if __name__ == '__main__':
	User.create_authors()
