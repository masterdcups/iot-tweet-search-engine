from joblib import dump, load
import os
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class ModelPrediction:
    path_gend_mod = 'gender_model.joblib'
    path_sent_mod = 'sentiment_model.joblib'
    path_coun_mod = 'country_model.joblib'

    def __init__(self, dir_path='saved_models', corpus_path='truc.txt'):
        # truc.txt est un fichier test où il y a les 10eres lignes du corpus
        self.dir_path = dir_path
        self.map = None
        self.corpus_path = corpus_path

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def gender_model(self, file_path=path_gend_mod):
        """Method to create model prediction user's gender using a SVM classifier"""
        full_path = os.path.join(self.dir_path, file_path)

        if not os.path.exists(full_path):

            if self.map is None:
                self.map = ModelPrediction.parsing_corpus(self.corpus_path)

            X = self.map['Vector']
            y = self.map['Gender']

            model = svm.SVC(kernel='linear')
            model.fit(X, y)

            dump(model, full_path)
        else:
            model = load(full_path)

        return model

    def sentiment_model(self, file_path=path_sent_mod):
        """Method to create model prediction user's sentiment using a CNN"""
        full_path = os.path.join(self.dir_path, file_path)

        if not os.path.exists(full_path):

            if self.map is None:
                self.map = ModelPrediction.parsing_corpus(self.corpus_path)

            X = self.map['Vector']
            y = self.map['Sentiment']

            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            clf.fit(X, y)

            dump(clf, full_path)
        else:
            clf = load(full_path)

        return clf

    def country_model(self, file_path=path_coun_mod):
        """Method to create model prediction user's country using a Naive Bayes network"""
        full_path = os.path.join(self.dir_path, file_path)

        if not os.path.exists(full_path):

            if self.map is None:
                self.map = ModelPrediction.parsing_corpus(self.corpus_path)

            X = self.map['Vector']
            y = self.map['Country']

            gnb = GaussianNB()
            gnb.fit(X, y)

            dump(gnb, full_path)
        else:
            gnb = load(full_path)

        return gnb

    @staticmethod
    def parsing_corpus(path):
        # Les lignes en commentaires devront être décommentées lorsque le corpus sera complété avec le texte des tweets et les vecteurs associés
        # Les deux lignes avant "fichier.close()" devront alors être supprimées
        map = {}
        with open(path, "r") as fichier:
            line = fichier.readline().replace('\n', '').split('\t')
            for key in line:
                map[key] = []
            map['Vector'] = []
            for line in fichier:
                tweet = line.replace('\n', '').split("\t")
                map['TweetID'] += [tweet[0]]
                map['Sentiment'] += [tweet[1]]
                map['TopicID'] += [tweet[2]]
                map['Country'] += [tweet[3]]
                map['Gender'] += [tweet[4]]
                # map['URLs'] += tweet[5:-2]
                # map['Text'] += tweet[-2]
                # map['Vector'] += tweet[-1]
                map['URLs'] += [tweet[5:]]
                map['Vector'] += [np.zeros(300)]
        fichier.close()
        return map


if __name__ == '__main__':
    model = ModelPrediction()
    print(model.gender_model().predict(np.zeros(300).reshape(1, -1)))
    print(model.sentiment_model().predict(np.zeros(300).reshape(1, -1)))
    print(model.country_model().predict(np.zeros(300).reshape(1, -1)))
