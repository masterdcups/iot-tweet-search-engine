import os

import numpy as np

from keras.engine.saving import model_from_yaml
from keras.layers import Embedding, Input, Dense, Flatten, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

import sys

sys.path.append('../')
from definitions import ROOT_DIR
from parser2 import Parser2


class GMF_model:

    def __init__(self):
        return

    def get_model(self, num_users, num_items, latent_dim, regs=[0, 0]):
        """
        Method to generate the GMF model
        :param num_users: number of users
        :param num_items: number of tweets
        :param latent_dim: Embedding size
        :param regs: Regularization for user and item embeddings
        :return: GMF model
        """
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                      init='normal', W_regularizer=l2(regs[0]), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                      init='normal', W_regularizer=l2(regs[1]), input_length=1)

        # Crucial to flatten an embedding vector
        user_latent = Flatten()(MF_Embedding_User(user_input))
        item_latent = Flatten()(MF_Embedding_Item(item_input))

        # Element-wise product of user and item embeddings
        predict_vector = concatenate([user_latent, item_latent])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

        model = Model(input=[user_input, item_input], output=prediction)

        return model

    def get_train_instances(self, train, num_negatives, num_tweets):
        """
        Method to train the instances
        :param train: part of the corpus
        :param num_negatives: number of negative instances to pair with a positive instance
        :param num_tweets: number of tweet
        :return: user_input, item_input, labels
        """
        user_input, item_input, labels = [], [], []

        for index, tweet in train.iterrows():
            u = tweet.User_ID_u
            i = tweet.TweetID_u
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_tweets)
                while (u, j) in train[['User_ID_u', 'TweetID_u']]:
                    j = np.random.randint(num_tweets)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels

    def prediction(self):
        """
        Method to predict the tweets
        """
        f = open('train_0.yaml', 'r')
        model = model_from_yaml(f.read())
        f.close()
        model.load_weights('train_0.model')

        corpus = Parser2.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v3.tsv'),
        										  categorize=True)

        train, test = train_test_split(corpus, test_size=0.2)

        num_tweets = corpus.TweetID_u.max() + 1
        num_negatives = 20

        for index, tweet in test.iterrows():
            u = tweet.User_ID_u
            items = []
            #print('utilisateur: ', u)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_tweets)
                while (u, j) in train[['User_ID_u', 'TweetID_u']]:
                    j = np.random.randint(num_tweets)
                # user_input.append(u)
                items.append(j)

            #print('negatives tweets de ', u, ': ', items)

            users = np.full(len(items), u, dtype='int32')
            predictions = model.predict([users, np.array(items)], batch_size=100, verbose=0)
            #print('predictions: ', predictions)

    def training(self, train, num_negatives, num_tweets, model):
        """
        Method to train
        :param train: part of the corpus
        :param num_negatives: Number of negative instances to pair with a positive instance
        :param num_tweets: number of tweet
        :param model: GMF model
        """
        batch_size = 256
        model_out_file = 'train_'
        epoch = 1

        # Generate training instances
        user_input, item_input, labels = self.get_train_instances(train, num_negatives, num_tweets)
        #print(user_input)
        #print(item_input)
        #print(labels)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

        model.save_weights(model_out_file + str(epoch) + '.model', overwrite=True)

        f = open(model_out_file + str(epoch) + '.yaml', 'w')
        yaml_string = model.to_yaml()
        f.write(yaml_string)
        f.close()

    def model_creation(self):
        """
        Method to create the GMF model
        """
        corpus = Parser2.parsing_iot_corpus_pandas(os.path.join(ROOT_DIR, 'corpus/iot-tweets-vector-v3.tsv'),
                                                 categorize=True)

        num_negatives = 4  # Number of negative instances to pair with a positive instance.
        regs = [0, 0]  # Regularization for user and item embeddings.
        num_factors = 8  # Embedding size
        learning_rate = 0.001


        num_users = corpus.User_ID_u.max() + 1
        num_tweets = corpus.TweetID_u.max() + 1

        #print(num_users, 'users')
        #print(num_tweets, 'tweets')

        # corpus = corpus[corpus.User_ID >= 0]
        train, test = train_test_split(corpus, test_size=0.2)

        # Build model
        model = self.get_model(num_users, num_tweets, num_factors, regs)

        # Compiling model
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

        self.training(train, num_negatives, num_tweets, model)

if __name__ == '__main__':
    GMF = GMF_model()
    #GMF.model_creation()
    GMF.prediction()