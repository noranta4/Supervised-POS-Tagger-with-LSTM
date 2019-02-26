from homework2 import AbstractLSTMPOSTagger, ModelIO
import Word_embedding
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

class LSTMPOSTagger(AbstractLSTMPOSTagger):
    def __init__(self, model, resource_dir=None):
        self._model = model
        self._resource_dir = resource_dir
    def load_resources(self):
        # word2vec setup with google news data
        print('loading word2vec model...')
        self.vecmodel = gensim.models.KeyedVectors.load_word2vec_format(self._resource_dir + 'GoogleNews-vectors-negative300.bin', binary=True)
        # self.vecmodel_ita = gensim.models.KeyedVectors.load(self._resource_dir + "it/it.bin") # DECOMMENT IF MULTILINGUAL
        pass
    def predict(self, sentence):
        """
        predict the pos tags for each token in the sentence.
        :param sentence: a list of tokens.
        :return: a list of pos tags (one for each input token).
        """
        tags = ['PART', 'CONJ', 'VERB', 'PUNCT', 'AUX', 'PRON', 'NUM', 'ADV', 'SCONJ', 'INTJ', 'DET', 'SYM',
                'PROPN', 'ADP', 'NOUN', 'X', 'ADJ']
        max_length = self._model.input_shape[1] # obtain admissible length of a sentence (max_length) from the model
        missing_spaces = max_length - (len(sentence) % max_length) # make the number of element of the sentence divisible by max_length filling missing words with '-'
        original_sentence = list(sentence)
        for i in range(missing_spaces):
            sentence.append('-')
        if len(original_sentence) > max_length: # re-add to the sentence the last max_length words
            for i in reversed(range(1, max_length + 1)):
                sentence.append(original_sentence[-i])

        sentence_vec = [] # list (sentence) of the word vectors
        for word in sentence:
            try:
                sentence_vec.append(self.vecmodel_ita[word])
            except:
                sentence_vec.append(Word_embedding.word2vec(self.vecmodel, word)) # append the word2vec vector
            """
            see Word_embedding.py for details about the manage of the missing word in the word2vec model
            """
        chunks = [sentence_vec[x:x + max_length] for x in range(0, len(sentence_vec), max_length)] # split the sentence in subsentences of length == max_length
        if len(original_sentence) > max_length:
            chunks.pop(-2) # remove the chunk with '-'
        chunks = np.array(chunks) # conversion to numpy array
        result = self._model.predict(chunks) # prediction of POSTags
        result_tags = [] # build a list with the predicted POStag
        for j in range(len(result)):
            for i in range(len(result[j])):
                result_tags.append(tags[int(list(result[j][i]).index(max(result[j][i])))])
        if len(original_sentence) > max_length:
            for i in range(missing_spaces): # exclude repeated words
                sentence.pop(-(max_length - missing_spaces + 1))
                result_tags.pop(-(max_length - missing_spaces + 1))
            for i in range(max_length): # exclude the added '-'
                sentence.pop(-(max_length - missing_spaces + 1))
        else:
            for i in range(missing_spaces): # exclude the added '-'
                sentence.pop()
                result_tags.pop()
        # print(sentence)
        # print(result_tags)
        return result_tags

# TESTING THE CLASS ####################################################################################################

# model = ModelIO().load('LSTM_POStag.keras.model2')
# prova = LSTMPOSTagger(model, resource_dir="D:/Universita/Intelligenza_Artificiale_e_Robotica/Natural_language_processing/GoogleNews-vectors-negative300.bin")
# prova.load_resources()
# print(prova.predict(['this', 'is', 'an', 'easy', 'test', '.']))
# print('gold : ' + str(['PRON', 'VERB', 'DET', 'ADJ', 'NOUN', 'PUNCT']))


