import os

import networkx as nx
import numpy as np

from definitions import ROOT_DIR
from parser import Parser
from prediction_profile import PredictionProfile
from topics_classifier import TopicsClassifier


class User:
	user_fname = os.path.join(ROOT_DIR, 'saved_models/users_profile.tsv')
	author_fname = os.path.join(ROOT_DIR, 'saved_models/authors_profile.tsv')
	user_graph_path = os.path.join(ROOT_DIR, 'corpus/author_graph.net')

	def __init__(self, id=None, nb_click=0, vector=np.zeros(300), localisation='', gender='', emotion='',
				 topic_vector=np.asarray([]), centrality=0., vec_size=300):

		self.id = None

		if id is None:
			self.id = self.next_id()
			self.vec = np.zeros(vec_size)
			self.nb_click = 0
			self.localisation = ''
			self.gender = ''
			self.emotion = ''
			self.topic_vector = np.asarray([])
			self.centrality = 0.
		else:
			self.id = id
			self.vec = vector
			self.nb_click = nb_click
			self.localisation = localisation
			self.gender = gender
			self.emotion = emotion
			self.topic_vector = topic_vector
			self.centrality = centrality

		self.prediction_profile = None
		self.topic_classifier = None

	def set_prediction_profile(self, pp):
		self.prediction_profile = pp

	def set_topic_classifier(self, tpc):
		self.topic_classifier = tpc

	def get_prediction_profile(self):
		if self.prediction_profile is None:
			self.prediction_profile = PredictionProfile()

		return self.prediction_profile

	def get_topic_classifier(self):
		if self.topic_classifier is None:
			self.topic_classifier = TopicsClassifier()

		return self.topic_classifier

	def predict_profile(self):
		"""
		Call all the predictions models to fill the localisation, gender, etc
		:return:
		"""

		self.localisation = self.get_prediction_profile().country_prediction(self.vec)
		self.gender = self.get_prediction_profile().gender_prediction(self.vec)
		self.emotion = self.get_prediction_profile().sentiment_prediction(self.vec)
		self.topic_vector = self.get_topic_classifier().predict(self.vec.reshape(1, -1))[0]

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

		for j in range(1, len(contents)):
			items = contents[j].split('\t')
			id = int(items[0]) if type(self.id) is int else items[0]
			users_data[id] = j
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
		"""Load the user from the corresponding file of do nothing"""
		assert self.id is not None

		self.create_files()
		f = open(User.user_fname if type(self.id) is int else User.author_fname, "r")
		lines = f.readlines()
		for i in range(1, len(lines)):
			l = lines[i][:-1]
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

	def next_id(self):
		"""Get the max +1 id in the file"""
		self.create_files()
		f = open(User.user_fname, "r")
		contents = f.readlines()
		if len(contents) == 1:
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
		lines = file.readlines()
		for i in range(1, len(lines)):
			line = lines[i]
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
		if (type(self.id) is int or self.id is None) and not os.path.exists(User.user_fname):
			f = open(User.user_fname, 'w+')
			f.write('User_Name\tNbClick\tVector\tLocalisation\tGender\tEmotion\tTopicVector\tCentrality\n')
			f.close()

		if type(self.id) is str and not os.path.exists(User.author_fname):
			f = open(User.author_fname, 'w+')
			f.write('User_Name\tNbClick\tVector\tLocalisation\tGender\tEmotion\tTopicVector\tCentrality\n')
			f.close()

	@staticmethod
	def create_authors(corpus):
		"""
		Generate the authors_profile.tsv file
		To perform just ONE time
		:type corpus: pandas.DataFrame
		:return:
		"""

		tpc = TopicsClassifier(pd_corpus=corpus)
		pp = PredictionProfile(pd_corpus=corpus)

		for index, tweet in corpus.iterrows():
			u = User(tweet.User_Name)
			u.load()
			u.update_profile(tweet.Vector, predict=False)
			u.save()

		graph = User.load_graph()
		centralities = nx.eigenvector_centrality(graph)
		for author in User.get_all_authors():
			author.centrality = centralities[author.id] if author.id in centralities else 0.
			author.set_prediction_profile(pp)
			author.set_topic_classifier(tpc)
			author.predict_profile()
			author.save()
		return

	@staticmethod
	def load_graph(filename=user_graph_path):
		return nx.DiGraph(nx.read_adjlist(filename))


if __name__ == '__main__':
	corpus = Parser.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v31.tsv'))
	print('Corpus Loaded')
	User.create_authors(corpus)
