#!/usr/bin/env python
# coding: utf-8


#=======================  import des libraries

import nltk
import string
import spacy
import pickle
import unidecode
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from joblib import load


#=======================  gestion des warnings

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#=======================  définition des fonctions

def lemmatise_text(text):
    lst_lematised = [token.lemma_ for token in nlp(text)] 
    return ' '.join(lst_lematised).lower()


def stem_text(text):
    lst_stemmerised = [stemmer.stem(token) for token in word_tokenize(text)]    
    return ' '.join(lst_stemmerised)


def substitute_punctuation(text):
    return ' '.join(text.replace("'", ' ').translate(str.maketrans('', '', string.punctuation)).split())


def supp(text):
    return text.replace("«", "").replace("’", "").replace("•", "").replace("®", "")


def supp_sw(text):
    return ' '.join([token.text for token in nlp(text) if not token.text in sw])


def supp_deb(chaine):
    chaine = re.sub(' +', ' ', chaine).lstrip()
    return re.sub(' +', ' ', chaine[0].replace(",", "").replace(";", "") + chaine[1:]).lstrip()
    

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


def generate_text(model_generation_text, start_string, num_generate=100): 
  # Evaluation step (generating text using the learned model)
  # num_generate = Number of characters to generate

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model_generation_text.reset_states()
  for i in range(num_generate):
      predictions = model_generation_text(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


#=======================  programme principal


# -- défintions de variables

stemmer = SnowballStemmer('french')
nlp = spacy.load('fr_core_news_sm')

sw = nltk.corpus.stopwords.words('french')
sw += ['être', 'avoir']


dic_code_theme = {"Préparer mon séjour": 1,
                  "Réserver et payer": 2,
                  "Gérer ma réservation": 3,
                  "Mon séjour": 4,
                  "Assurances": 5}
dic_decode_theme = {val: key for key, val in dic_code_theme.items()}



# -- défintions des modèles

# classif theme
vectoriser_theme = load('vectorizer_classif_theme.joblib')
classifier_theme = load('model_classif_theme.joblib')      #A VOIR AVEC ENO

# classif domaine
classifier_domaine = load('model_classif_domaine.joblib')

# import faq pour similarité
faq = pd.read_pickle('df_classif_similarity.pkl')
faq['tokens'] = faq['question_clean'].apply(nlp)

# generation réponse origniale
with open('model_generation_text.json', 'r') as json_file :
    loaded_model_json = json_file.read()

model_generation_text = tf.keras.models.model_from_json(loaded_model_json)
model_generation_text.load_weights("model_generation_text.h5")
checkpoint_dir = './training_checkpoints'
tf.train.latest_checkpoint(checkpoint_dir)
embedding_dim = 256
rnn_units = 1024    # Number of RNN units
model_generation_text = build_model(205, embedding_dim, rnn_units, batch_size=1)
model_generation_text.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model_generation_text.build(tf.TensorShape([1, None]))
# Chargement de char2idx et de idx2char
with open('char2idx.pkl','rb') as f:
    char2idx = pickle.load(f)
with open('idx2char.pkl','rb') as f:
    idx2char = pickle.load(f)
# Fonction de génération d'une réponse originale
input_eval = [char2idx[s] for s in u"bonjour"]
input_eval = tf.expand_dims(input_eval, 0)





# -- lancement du bot

print("Pour quitter, taper 'quit'.")

quest_user = input("Entrée user : ")

while quest_user != "quit":
    quest_user_clean = supp(substitute_punctuation(stem_text(lemmatise_text(quest_user))))
    X_quest_user = pd.Series(quest_user_clean)
    X_quest_user_clean_vectorized_tfidf = vectoriser_theme.transform(X_quest_user)
    domaine_quest_user = classifier_domaine.predict(X_quest_user_clean_vectorized_tfidf)
    if domaine_quest_user == 1:
        XX_quest_user = X_quest_user_clean_vectorized_tfidf.toarray().reshape(X_quest_user_clean_vectorized_tfidf.shape[0],1,
                                                X_quest_user_clean_vectorized_tfidf.shape[1])
        pred_proba = classifier_theme.predict(XX_quest_user)
        idx = np.argmax(pred_proba, axis=-1)
        YY_pred = np.zeros( pred_proba.shape )
        YY_pred[ np.arange(YY_pred.shape[0]), idx] = 1
        theme_quest_user = list(YY_pred[0]).index(1) +1
        print(theme_quest_user)
        faq_theme = faq[faq.theme == dic_decode_theme[theme_quest_user]][["question", 'reponse', 'tokens']]
        quest_user_clean = supp_sw(supp(substitute_punctuation(stem_text(lemmatise_text(quest_user)))))
        quest_user_clean_tokens = nlp(quest_user_clean)
        lst_similarity = [quest_user_clean_tokens.similarity(token) for token in faq_theme.tokens]
        print('\n', faq_theme.iloc[np.asarray(lst_similarity).argmax()].reponse, sep = 'Réponse bot : ')
    else:
        quest_user_clean = unidecode.unidecode(quest_user).replace("?","")
        gene0 = generate_text(model_generation_text, start_string=quest_user_clean)[len(quest_user_clean):]
        print('\n', supp_deb([p for p in gene0.split('.') if p!='' and len(p)>10][0]), sep = 'Réponse bot : ')
    quest_user = input("Entrée user : ")


