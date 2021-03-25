import numpy as np
import tensorflow as tf
import io, re, unidecode, pickle, json
#import nltk.translate.bleu_score as bleu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import Encoder, BahdanauAttention, Decoder

def tokenizeText(listOfStentences):
    '''
    Permet de transformer un mot en nombre ainsi que fabriquer un dictionnaire

    listOfStentences: Tableau de phrases
    '''
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='' , oov_token='<unk>')
    tokenizer.fit_on_texts(listOfStentences)
    tensor = tokenizer.texts_to_sequences(listOfStentences)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer

def loadAndCreateDataset(path, nombreLignes):
    '''
    Cette fonction permet lire le contenu d'une dataset, ajouter <start> et <end> en début et fin de chaque phrase et supprime tous les caractères qui pourraient être dérangent pour l'apprentissage

    path: Chemin vers le fichier de la dataset
    nombreLignes: nombre de lignes que la fonction renvoie
    '''
    #Le programme créer un tableau du nombre de lignes que le fichier contient et il supprime les potentiels espaces en début/fin de ligne.
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    #Permet de créer un tupple entre les phrases "d'entrée" et celle de "sortie" et prend les nombreLignes premières lignes du premier tableau.
    word_pairs = [['<start> ' + preprocessSentences(w) + ' <end>' for w in l.split('\t')]  for l in lines[:nombreLignes]]
    #Permet de séparer le tupple en 2 tableaux
    inputTensor, targetTensor = zip(*word_pairs)
    print(inputTensor[1])
    print(targetTensor[1])
    #Tokenizer le texte permet d'obtenir celui-ci uniquement en nombre (Traduction de mot à nombre) et obtenir le dictionnaire de ce tensor
    inputTensor, inputVocab = tokenizeText(inputTensor)
    targetTensor, targetVocab = tokenizeText(targetTensor)
    #Permet d'avoie la longueur des 2 plus longues phrases
    inputSentencesSize = inputTensor.shape[1]
    targetSentencesSize = targetTensor.shape[1]
    #Le programme retourne les informations nécessaires.
    return inputTensor, inputVocab, inputSentencesSize, targetTensor, targetVocab, targetSentencesSize

def preprocessSentences(line):
  '''
  Cette fonction utilise des expressions régulières afin d'enlever les éléments perturbateurs d'une phrase (Cela permet une simplification de la langue). 

  line: Phrase
  '''
  line = re.sub(r"([?.!,¿])", r" \1 ", line)
  line = re.sub(r'[" "]+', " ", line)
  line = unidecode.unidecode(line) 
  line = re.sub(r"[^a-zA-Z?.!,¿]+", " ", line)
  return line.lower()

def statsDataPreprocessed(inputTensor, targetTensor, inputVocab, targetVocab):
    '''
    Affiche dans le terminal quelques infos sur la taille maximale d'une phrase de la langue d'entrée et de sortie

    ***Tensor: Tensor de nombre (Phrases qui ont été indexées)
    ***Vocab: Object de type tokenizer
    '''
    print(targetTensor[1])
    print(inputTensor[1])
    print("The longest input sentence:", inputTensor.shape[1])
    print("The longest output sentence:", targetTensor.shape[1])
    print("Input vocabulary size:", len(inputVocab.word_index))
    print("Output vocabulary size:", len(targetVocab.word_index))

def createDataset(inputTensor, targetTensor, batchSize, bufferSize):
    '''
    Permet de créer la dataset d'entraînement

    ***Tensor: Tensor de nombre (Phrases qui ont été indexées)
    batchSize: Nombre de tableau de phrases
    bufferSize: Nombre de phrases présentent dans inputTensor
    '''
    dataset = tf.data.Dataset.from_tensor_slices(
        (inputTensor, targetTensor)).shuffle(bufferSize) #Création une dataset avec des phrases tirées aléatoirement dans inputTensor et targetTensor
    dataset = dataset.batch(batchSize, drop_remainder=True) #Permet de faire batchSize fois
    inputBatch, targetBatch = next(iter(dataset)) #Permet de former 2 tableaux l'un avec les phrases d'entrées et l'autre avec celle de sortie 
    return inputBatch, targetBatch, dataset

def lossFunction(real, pred, loss_object):
  '''
  Permet de calculer l'erreur de l'entraînement
  '''
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def trainStep(inp, targ, enc_hidden, encoder, decoder, targetVocab, lossObject, optimizer, batchSize):
  '''
  Entraîne phrase par phrase et retourne l'erreur totale du batch

  inp: Phrase d'entrée
  targ: Phrase de sortie
  enc_hidden: état caché de l'encodeur
  encoder: Object de l'encoder
  decoder:Object du decoder
  targetVocab: Objet de type tokenizer
  lossObject: Objet qui permet le calcul de perte
  optimizer: Adam
  batchSize: Nombre de batch
  '''
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targetVocab.word_index['<start>']] * batchSize, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += lossFunction(targ[:, t], predictions, lossObject)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  #Descente du gradiant
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def evaluate(sentence, vocabTargetSize, vocabInputSize, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):
  '''
  Permet de traduire une phrase

  sentence: Phrase à traduire
  targetTensor: Taille du vocabulaire de la phrase traduite 
  inputTensor: Taille du vocabulaire de la phrase à traduire
  inputSize: Taille de la plus grande phrase présente dans la dataset de la phrase traduite
  targetSize: Taille de la plus grande phrase présente dans la dataset de la phrase à traduire
  inputVocab: Tokenizer de l'input
  targetVocab: Tokenizer de target
  units: Nombre de phrases dans chaque batch
  encoderModel: l'encoder de l'IA
  decoder: le le decoder de l'IA
  '''

  #Matrice de 0 de la taille de targetTensorTrain lignes et inputTensorTrain colonnes
  attention_plot = np.zeros((vocabTargetSize, vocabInputSize))

  #Enlève les éléments indésirables et ajoute les balises <start> et <end>
  sentence = '<start> ' + preprocessSentences(sentence) + ' <end>'

  #Transforme les mots en nombre, si le mot n'est pas dans le dictionnaire, il est remplacé par la valeur da la balise <unk> qui permet de traduire une phrase sans connaître tous les mots
  inputs = [inputVocab.word_index[i] if i in inputVocab.word_index   else inputVocab.word_index['<unk>'] for i in sentence.split(' ')]
  #Reshape le tableau afin qu'il fasse la longueur de la plus longue phrase du même langage
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=inputSize,
                                                         padding='post')

  #Conversion du tableau en tensor                             
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  #Matrice de 0 de la taille d'une ligne et units colonnes
  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoderModel(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targetVocab.word_index['<start>']], 0)

  #Tant que t en inférieur à targetSize
  for t in range(int(targetSize)):
    
    #On réalise une prédiction et on récupère les poids du mécanisme d'attention
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    #Stocker les poids d'attention pour les tracer plus tard
    attention_weights = tf.reshape(attention_weights, (-1, ))
    #attention_plot[t] = attention_weights.numpy() J'AI UN PROBLEME ICI QUE JE N'AI PAS PRIT LE TEMPS DE CORRIGER

    #Récupère la meilleure prédiction 
    predicted_id = tf.argmax(predictions[0]).numpy()

    #Conversion de l'id prédit vers le mot qui lui est associé dans le dictionnaire et ajoute un espace pour fabriquer la phrase
    result += targetVocab.index_word[predicted_id] + ' '

    #Lorsque le modèle prédit '<end>', le programme retourne la traduction ainsi que la phrase à traduire
    if targetVocab.index_word[predicted_id] == '<end>':
      return result, sentence#, attention_plot

    #Sinon l'identifiant prédit est réinjecté dans le modèle.
    dec_input = tf.expand_dims([predicted_id], 0)
  
  #Et on retourne quand même le résultat
  return result, sentence#, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
  '''
  Permet de tracer l'attention apportée par l'IA sur chaque mot traduit par rapport à ceux à traduire

  attention: résultat du mécanisme d'attention pour cette traduction
  sentence: Phrase à traduire
  predicted_sentence: Phrase traduite
  '''
  #Création de la figure de 10*10
  fig = plt.figure(figsize=(10,10))

  #La première étant le nombre de lignes de la grille, la deuxième étant le nombre de colonnes de la grille et la troisième étant la position à laquelle le nouveau sous-plot doit être placé.
  ax = fig.add_subplot(1, 1, 1)
  #Afficher un tableau sous forme de matrice
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  #La phrase à traduire comme titre de l'axe x 
  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  #La phrase à traduire comme titre de l'axe y
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  #Echelle de 1 en 1
  ax.xaxis.set_major_locator((ticker.MultipleLocator(1)))
  ax.yaxis.set_major_locator((ticker.MultipleLocator(1)))

  #Afficher le graphique
  plt.show()

def translate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):
  '''
  Permet de traduire et de tracer l'attention de la traduction

  sentence: Phrase à traduire
  targetTensor: Taille du vocabulaire de la phrase traduite 
  inputTensor: Taille du vocabulaire de la phrase à traduire
  inputSize: Taille de la plus grande phrase présente dans la dataset de la phrase traduite
  targetSize: Taille de la plus grande phrase présente dans la dataset de la phrase à traduire
  inputVocab: Tokenizer de l'input
  targetVocab: Tokenizer de target
  units: Nombre de phrases dans chaque batch
  encoderModel: l'encoder de l'IA
  decoder: le le decoder de l'IA
  '''
  # result, sentence, attention_plot = evaluate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize,inputVocab, targetVocab, units, encoderModel, decoder)
  #Traduit la phrase
  result, sentence = evaluate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize,inputVocab, targetVocab, units, encoderModel, decoder)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

#Même fonction que evaluate sans le calcul de l'attention
def evaluateAPI(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):

  sentence = '<start> ' + preprocessSentences(sentence) + ' <end>'
  print(sentence)
  inputs = [inputVocab.word_index[i] if i in inputVocab.word_index else inputVocab.word_index['<unk>'] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=inputSize,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoderModel(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targetVocab.word_index['<start>']], 0)

  for t in range(targetTensorTrain):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targetVocab.index_word[predicted_id] + ' '

    if targetVocab.index_word[predicted_id] == '<end>':
      resultFind = ''
      for word in result.split(' '):
        if(word != '<end>'):
          resultFind += word + ' '
      return resultFind, sentence

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence

#Pareil que translate sans l'affichage de l'attention
def translateAPI(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):
  result, sentence = evaluateAPI(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder)
  return result


def createNetworkAndApplyCheckPointsAndConfig(configName):
  '''
  Permet de créer un réseau selon les caractéristiques présente dans le fichier config et appliquer les poids présents dans le répertoire de l'entraînement

  configName: nom du répertoire où le fichier de configuration et les poids sont enregistrés
  '''
  checkpointAndConfigAndVocabsDirectory = './outputs/{}/'.format(configName)
  
  #On lit le fichier de configuration, qui contient des informations clés pour réaliser des prédictions
  configTrain = json.load(open(checkpointAndConfigAndVocabsDirectory + 'config.json', 'r', encoding='UTF-8'))
  batchSize = 1 #Le programme n'a qu'un(e) seul paragraphe/seule phrase à traduire par prédiction
  
  #Création de l'encoder et du decoder selon les paramètres précédemment chargés
  encoder = Encoder(configTrain['vocabInputSize'], configTrain['embeddingDim'], configTrain['units'], batchSize)
  decoder = Decoder(configTrain['vocabTargetSize'], configTrain['embeddingDim'], configTrain['units'], batchSize)
  #On définit l'optimizer que j'utilise pour le projet
  optimizer = tf.keras.optimizers.Adam()
  #Il charge les poids de l'IA entraîné dans ce réseau
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
  manager = tf.train.CheckpointManager(checkpoint, checkpointAndConfigAndVocabsDirectory, max_to_keep=3)
  checkpoint.restore(manager.latest_checkpoint)
  #Il charge le tokenizer sauvegardé à la fin de l'apprentissage afin d'avoir le même dictionnaire
  with open( checkpointAndConfigAndVocabsDirectory + 'inputVocab.pickle', 'rb') as handle:
      inputVocab = pickle.load(handle)
  with open(checkpointAndConfigAndVocabsDirectory + 'targetVocab.pickle', 'rb') as handle:
      targetVocab = pickle.load(handle)

  return encoder, decoder, inputVocab, targetVocab, configTrain