import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class Prediction:
    dict = {}

    def __init__(self, path):  # Notre méthode constructeur
        """Pour l'instant, on ne va définir qu'un seul attribut"""
        self.path = path

    def parsingCorpus(self, path):
        with open(path2, "r") as fichier:
            line = fichier.readline().replace('\n', '').split('\t')
            for key in line:
                dict[key] = []
            dict['Vecteur'] = []
            for line in fichier:
                tweet = line.replace('\n', '').split("\t")
                dict['TweetID'] += [tweet[0]]
                dict['Sentiment'] += [tweet[1]]
                dict['TopicID'] += [tweet[2]]
                dict['Country'] += [tweet[3]]
                dict['Gender'] += [tweet[4]]
                # dict['URLs'] += tweet[5:-2]
                # dict['Texte'] += tweet[-2]
                # dict['Vecteur'] += tweet[-1]
                dict['URLs'] += [tweet[5:]]
                dict['Vecteur'] += [np.zeros(300)]
        fichier.close()

    @property
    def genderPrediction(self):
        X = dict['Vecteur']
        y = dict['Gender']

        # classifieur SVM
        model = svm.SVC(kernel='linear')
        model.fit(X, y)

        return model.predict(np.zeros(300).reshape(1, -1))

    print(genderPrediction)

    @property
    def sentimentPrediction(self):
        X = dict['Vecteur']
        y = dict['Sentiment']

        # CNN - reseau de neurones profonds
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(X, y)

        return clf.predict(np.zeros(300).reshape(1, -1))

    @property
    def countryPrediction(self):
        X = dict['Vecteur']
        y = dict['Country']

        # NaiveBayes
        gnb = GaussianNB()
        gnb.fit(X, y)

        return gnb.predict(np.zeros(300).reshape(1, -1))




    print(sentimentPrediction(dict))
