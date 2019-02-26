from homework2 import AbstractPOSTaggerTrainer
import Word_embedding
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed, Bidirectional


class POSTaggerTrainer(AbstractPOSTaggerTrainer):
    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir

    def load_resources(self):
        # word2vec setup with google news data
        print('loading word2vec model...')
        self.vecmodel = gensim.models.KeyedVectors.load_word2vec_format(self._resource_dir  + 'GoogleNews-vectors-negative300.bin', binary=True)
        # self.vecmodel_ita = gensim.models.KeyedVectors.load(self._resource_dir + "it/it.bin") # DECOMMENT IF MULTILINGUAL
        pass

    def train(self, training_path):
        """
        Train the keras model from the training data.

        :param training_path: the path to training file
        :return: the keras model of type Sequential
        """

        # fix random seed for reproducibility
        np.random.seed(7)

# IMPORTING DATA #######################################################################################################
        print('importing data...')
        max_length = 6  # max sentence lenght

        with open(training_path, encoding="utf8") as inputfile:
            X_train = [] # list (training file) of list (sentences) of vectors (300dim word2vec vectors)
            y_train = [] # list (training file) of list (sentences) of tags (17dim one-hot-encoding POStags vectors)
            X_train_words = [] # list (training file) of list (sentences) of strings (words)
            y_train_words = [] # list (training file) of list (sentences) of strings (POStags of words)

            sentence_vec = [] # list (sentences) of vectors (300dim word2vec vectors)
            sentence_tags = [] # list (sentences) of tags (17dim one-hot-encoding POStags vectors)
            sentence_words = [] # list (sentences) of strings (words)
            sentence_words_tags = [] # list (sentences) of strings (POStags of words)

            tags = ['PART', 'CONJ', 'VERB', 'PUNCT', 'AUX', 'PRON', 'NUM', 'ADV', 'SCONJ', 'INTJ', 'DET', 'SYM',
                    'PROPN', 'ADP', 'NOUN', 'X', 'ADJ']

            for line in inputfile:
                actual_line = line.split('\t')
                if line[0] != '#' and line[0] != '\n' and '-' not in actual_line[0]: # if the line has an instance
                    sentence_words.append(actual_line[1]) # append the word
                    sentence_words_tags.append(actual_line[3]) # append the POStag
                    one_hot_tag = [False for i in range(17)] # generate the corresponding one-hot-encoding vector of the POStag
                    one_hot_tag[tags.index(actual_line[3])] = True
                    sentence_tags.append(one_hot_tag) # append the one-hot-encoding vector of the POStag
                    try:
                        sentence_vec.append(self.vecmodel_ita[actual_line[1]])
                    except:
                        sentence_vec.append(Word_embedding.word2vec(self.vecmodel, actual_line[1])) # append the word2vec vector
                    """
                    see Word_embedding.py for details about the manage of the missing word in the word2vec model
                    """

                elif line[0] == '\n': # if the sentence is over
                    if len(sentence_vec) > max_length: # if the sentence is longer than the length admitted in the model (max_length)
                        # append to the training data all the sub-sentences of length == maximum_length
                        X_train.extend([sentence_vec[x:x + max_length] for x in range(0, len(sentence_vec) - max_length + 1)])
                        y_train.extend([sentence_tags[x:x + max_length] for x in range(0, len(sentence_tags) - max_length + 1)])
                        X_train_words.extend([sentence_words[x:x + max_length] for x in range(0, len(sentence_words) - max_length + 1)])
                        y_train_words.extend([sentence_words_tags[x:x + max_length] for x in range(0, len(sentence_words_tags) - max_length + 1)])
                    sentence_words = [] # reset the sentence containers
                    sentence_vec = []
                    sentence_words_tags = []
                    sentence_tags = []

            X_train = np.array(X_train) # convert training data into numpy arrays
            y_train = np.array(y_train)

        print('Embedding done')

# DEFINITION OF THE MODEL AND TRAINING #################################################################################

        # print(X_train.shape) # check the shape of the training data
        # print(y_train.shape)

        batch_size = 180

        model = Sequential()
        model.add(Bidirectional( # bidirectional LSTM of 100 neurons
            LSTM(100, use_bias=True, activation='tanh', return_sequences=True, implementation=2),
            input_shape=(max_length, 300)))
        model.add(TimeDistributed( # a dense layer of 17 neurons to every temporal slice of the input (TimeDistributed)
            Dense(17, use_bias=True, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        print(model.summary()) # print model summary
        model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=2) # training
        model.save('model.keras') # save the model
        return model

# TESTING THE CLASS ####################################################################################################

# model = POSTaggerTrainer(resource_dir="D:/Universita/Intelligenza_Artificiale_e_Robotica/Natural_language_processing/")
# model.load_resources()
# model.train('en-ud-train.conllu')