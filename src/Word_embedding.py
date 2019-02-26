from random import random

def word2vec(vecmodel, word):
    try:
        return vecmodel[word]
    except: # if the word is not in the word2vec database
        '''
        words are vectors?
        nice, so exploit the operations defined on vector spaces:
        sum of vectors and multipication by a scalar
        '''
        if word == 'to':  # in the training data 'to' is 63% PART and 35% ADP, "n't" and "within" are respectively 100% PART and 100% ADP
            return (0.64 * vecmodel["n't"] + 0.36 * vecmodel['within'])
        elif word == 'and':  # "and" has about the same tag of "or"
            return (vecmodel['or'])
        elif word == 'a':  # "a" has about the same tag of "the"
            return (vecmodel['the'])
        elif word == 'of':  # "of" has about the same tag of "in"
            return (vecmodel['in'])
        elif (word == "'s") or (word == "`s"):  # in the training data "'s" is 61% PART and 32% VERB and 7% AUX, "n't", "found" and "could" are respectively 100% PART, VERB and AUX
            return (
                0.61 * vecmodel["n't"] + 0.32 * vecmodel['found'] + 0.07 * vecmodel['could'])
        elif (word == "e-mail") or (word == "OBSF") or (
                    word == ".doc"):  # all have the same tag of "email" (NOUN)
            return (vecmodel['email'])
        elif word == 'neighbours':
            return (vecmodel['neighbors'])
        elif word == 'travelling':
            return (vecmodel['traveling'])
        elif word == 'EnronOnline':
            return (vecmodel['Bill'])
    
        elif any((c in set('0123456789') for c in word)):
            return (vecmodel['##'])  # word2vec sends numbers in '#', here I put two '#' because one is often labeled as NOUN in the data
        elif (any((c in set('.,-")(?:!/;<[]') for c in word))) or ("'" in word and 's' not in word):
            return (vecmodel['*'])  # all PUNCT in '*'
        else:  # all remaining probably are SYM or X, in the training data we have 41% SYM and 59% X
            random_number = random()
            if random_number < 0.41:
                return (vecmodel['%'])  # SYM in '%'
            else:
                return (vecmodel['etc.'])  # X in 'etc.'