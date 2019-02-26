from homework2 import AbstractPOSTaggerTester
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



class POSTaggerTester(AbstractPOSTaggerTester):
    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir

    def load_resources(self):
        pass

    def test(self, lstm_pos_tagger, test_file_path):
        """
        Test the lstm_pos_tagger against the gold standard.

        :param lst_pos_tagger: an istance of AbstractLSTMPOSTagger that has to be tested.
        :param test_file_path: a path to the gold standard file.

        :return: a dictionary that has as keys 'precision', 'recall',
        'coverage' and 'f1' and as associated value their respective values.

        Additional info:
        - Precision has to be computed as the number of correctly predicted
          pos tag over the number of predicted pos tags.
        - Recall has to be computed as the number of correctly predicted
          pos tag over the number of items in the gold standard
        - Coverage has to be computed as the number of predicted pos tag over
          the number of items in the gold standard
        - F1 has to be computed as the armonic mean between precision
          and recall (2* P * R / (P + R))
        """

# IMPORTING DATA #######################################################################################################

        print('importing data...')
        max_length = 6  # max sentence lenght

        with open(test_file_path, encoding="utf8") as inputfile:
            X_test_words = [] # list (test file) of list (sentences) of strings (words)
            y_test_words = [] # list (test file) of list (sentences) of strings (POStags of words)

            sentence_words = []  # list (sentences) of strings (words)
            sentence_words_tags = []  # list (sentences) of strings (POStags of words)


            for line in inputfile:
                actual_line = line.split('\t')
                if line[0] != '#' and line[0] != '\n': # if the line has an instance
                    sentence_words.append(actual_line[1]) # append the word
                    sentence_words_tags.append(actual_line[3]) # append the POStag

                elif line[0] == '\n': # if the sentence is over
                    X_test_words.append(sentence_words)
                    y_test_words.append(sentence_words_tags)
                    sentence_words = [] # reset the sentence containers
                    sentence_words_tags = []

# TEST DATA #######################################################################################################

        # the model gives a prediction always, so P = R = F1 and C = 1
        f = open('pos_tagged_sentences1.txt', 'w')
        lstm_pos_tagger.load_resources()
        total = 0
        correct = 0
        y_pred = []
        y_true = []
        for sentence_index in range(len(X_test_words)):
            prediction = lstm_pos_tagger.predict(X_test_words[sentence_index])
            gold_standard = y_test_words[sentence_index]
            try:
                f.write(str(X_test_words[sentence_index]) + '\n')
            except UnicodeEncodeError:
                print(X_test_words[sentence_index])

            f.write(str(prediction) + '\n')
            y_pred.extend(prediction)
            y_true.extend(gold_standard)
            for i in range(len(prediction)):
                total += 1
                if prediction[i] == gold_standard[i]:
                    correct += 1
        P = float(correct)/total
        R = float(correct)/total
        F1 = 2 * P * R / (P + R)
        C = 1
        # cnf_matrix = confusion_matrix(y_true, y_pred, labels=['PART', 'CONJ', 'VERB', 'PUNCT', 'AUX', 'PRON', 'NUM', 'ADV', 'SCONJ', 'INTJ', 'DET', 'SYM', 'PROPN', 'ADP', 'NOUN', 'X', 'ADJ'])
        return {'precision': P, 'recall': R, 'coverage': C, 'f1': F1}
        pass

# TESTING THE CLASS ####################################################################################################

# model = ModelIO().load('LSTM_POStag.keras.model2')
# prova = LSTMPOSTagger(model, resource_dir="D:/Universita/Intelligenza_Artificiale_e_Robotica/Natural_language_processing/")
# tester = POSTaggerTester(resource_dir = "D:/Universita/Intelligenza_Artificiale_e_Robotica/Natural_language_processing/")
# tester.load_resources()
# print(tester.test(prova, 'en-ud-test.conllu'))





