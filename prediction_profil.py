import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class Prediction:
    """Class to predict users' sentiment, gender and country"""

    def __init__(self, map):
        """this class needs a dictionary with TweetID, Sentiment, TopicID, Country, Gender, URLs, Text, Vector"""
        self.map = map

    @property
    def gender_prediction(self):
        """Method to predict user's gender using a SVM classifier"""
        X = self.map['Vector']
        y = self.map['Gender']

        model = svm.SVC(kernel='linear')
        model.fit(X, y)

        return model.predict(np.zeros(300).reshape(1, -1))

    @property
    def sentiment_prediction(self):
        """Method to predict user's sentiment using a CNN"""
        X = self.map['Vector']
        y = self.map['Sentiment']

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(X, y)

        return clf.predict(np.zeros(300).reshape(1, -1))

    @property
    def country_prediction(self):
        """Method to predict user's country using a Naive Bayes network"""
        X = self.map['Vector']
        y = self.map['Country']

        gnb = GaussianNB()
        gnb.fit(X, y)

        return gnb.predict(np.zeros(300).reshape(1, -1))

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
    # truc.txt est un fichier test où il y a les 10eres lignes du corpus
    map = parsing_corpus('truc.txt')
    pred = Prediction(map)

    print(pred.gender_prediction)
