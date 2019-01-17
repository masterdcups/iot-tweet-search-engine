from joblib import dump, load
import os
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


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

    def tweak_hyperparameters(self, type_model):
        self.map = ModelPrediction.parsing_corpus(self.corpus_path)
        if type_model == "SVM":
            X = self.map['Vector']
            y = self.map['Gender']

            model = svm.SVC()
            param_grid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
            clf.fit(X, y)

            # Best parameter set
            print('Best parameters found:\n', clf.best_params_)
        else:
            X = self.map['Vector']
            y = self.map['Sentiment']

            model = MLPClassifier(max_iter=100)
            param_grid = {
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=3)
            clf.fit(X, y)

            # Best parameter set
            print('Best parameters found:\n', clf.best_params_)

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
    model.tweak_hyperparameters("SVM")
    model.tweak_hyperparameters("MLP")
    # print(model.gender_model())
    # print(model.sentiment_model())
    # print(model.country_model())
    # print(model.gender_model().predict(np.zeros(300).reshape(1, -1)))
    # print(model.sentiment_model().predict(np.zeros(300).reshape(1, -1)))
    # print(model.country_model().predict(np.zeros(300).reshape(1, -1)))
