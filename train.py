from argparse import Namespace #argparse est utile pour executé le programme avec des arguments
import os, time, pickle, json, random #Interaction avec l'OS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Le programme n'affiche que les problèmes de tensorflow (Permet une meilleure visibilité dans la console)

import tensorflow as tf
from libProject import loadAndCreateDataset, statsDataPreprocessed, createDataset, trainStep #Mes fonctions qui permettent une meilleure visibilité dans le code
from model import Encoder, BahdanauAttention, Decoder #Import du modèle de l'IA

def train(args: Namespace):
  '''
  Cette fonction permet l'entraînement de l'IA selon différents paramètres transmis lors de l'appel de celle-ci

  args: Renvoie tous les arguments nécessaires
  '''

  #Télécharge la dataset nommé fra-end.zip et l'extrait dans mon cas ici C:\Users\theof\.keras\datasets\ uniquement si celui-ci n'a pas déjà été téléchargé et extrait
  pathFile =  tf.keras.utils.get_file('fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
  extract=True)

  #Création de la première dataset
  inputTensor, inputVocab, inputSentencesSize, targetTensor, targetVocab, targetSentencesSize = loadAndCreateDataset(os.path.dirname(pathFile)+"/fra.txt", args.sentences_size)
  print('Datasets chargées')  
  
  #Quelques informations utiles sont affichées dans le terminal
  statsDataPreprocessed(inputTensor, targetTensor, inputVocab, targetVocab)

  epochSize = args.epoch
  batchSize = args.batch_size
  units = args.units
  bufferSize = len(inputTensor)
  embeddingDim = args.embedding_dim    
  
  #Création de la dataset pour l'entrainement
  inputBatch, targetBatch, dataset = createDataset(inputTensor, targetTensor, batchSize, bufferSize)   
  
  #On stocke la taille du vocabulaire dans une variable
  vocabInputSize = len(inputVocab.word_index)+1
  vocabTargetSize = len(targetVocab.word_index)+1

  #Création de l'encodeur avec certaines caractéristiques
  encoderModel = Encoder(vocabInputSize, embeddingDim, units, batchSize)
  #On met tous les hidden states à 0
  encodeHiddenLayers = encoderModel.initialize_hidden_state()
  #On récupère tous les hidden states de l'encoder
  encodedOutputLayers, encodeHiddenLayers = encoderModel(inputBatch, encodeHiddenLayers)
  
  print ('Encoder output shape: (batch size, sequence length, units) {}'.format(encodedOutputLayers.shape))
  print ('Encoder Hidden state shape: (batch size, units) {}'.format(encodeHiddenLayers.shape))
  
  #Création du mécanisme d'attention avec une taille de 10 couches
  attention_layer = BahdanauAttention(10)
  attention_result, attention_weights = attention_layer(encodeHiddenLayers, encodedOutputLayers)
  
  print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
  print("Attention weights shape: (batch size, sequence_length, 1) {}".format(attention_weights.shape))
  
  #Création du décodeur avec certaines caractéristiques
  decoder = Decoder(vocabTargetSize, embeddingDim, units, batchSize)
  sampleDecoderOutput, _, _ = decoder(tf.random.uniform((batchSize, 1)),encodeHiddenLayers, encodedOutputLayers)
  
  print ('Decoder output shape: (batch size, vocab size) {}'.format(sampleDecoderOutput.shape))

  #Définition de l'optimizer de l'entraînement nommé Adam qui repose sur un algorithme du gradient stochastique
  optimizer = tf.keras.optimizers.Adam()
  #Objet contient une méthode qui permet le calcule de la perte de l'entropie croisée entre les entrées et les prédictions.
  lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  
  #Ssi aucun nom de configuration n'est proposé dans les arguments
  if(args.config_name == None):
    #Le programme génère un nombre aléatoirement entre 0 et 9999
    configName = random.randint(0, 9999)
  #Sinon il copie la valeur de l'argument dans une variable
  else:
    configName = args.config_name
  
  #URL où sont stockés les checkpoints et le fichier de configuration de cet entraînement
  checkpointDirectory = './outputs/{}/'.format(configName)
  checkpoint_prefix = os.path.join(checkpointDirectory, "ckpt")

  #Objet qui va stocker les poids de l'entraînement selon l'optimiseur, l'encodeur et le décodeur
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                               encoder=encoderModel,
                               decoder=decoder)
  
  #Pour le nombre d'entraînements
  for epoch in range(epochSize):
      start = time.time()
      #Initialisation à 0 des hidden states de l'encodeur 
      enc_hidden = encoderModel.initialize_hidden_state()
      total_loss = 0     
      
      #Pour chaque ligne de la dataset, on récupère les phrases ainsi que leurs positions (Elles ont le même)  
      for (batch, (inp, targ)) in enumerate(dataset.take(len(inputTensor))):
        #Renvoie l'erreur
        batch_loss = trainStep(inp, targ, enc_hidden, encoderModel, decoder, targetVocab, lossObject, optimizer, batchSize)
        #On stocke l'erreur totale de l'entraînement
        total_loss += batch_loss      
        #Si l'entraînement est fini
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
      #Si la taille de l'entraînement est supérieur à 1
      if(epochSize > 1):
        #Le programme sauvegarde une fois sur deux le checkpoint
        if (epoch + 1) % 2 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)      
      #Sinon il sauvegarde le seul entraînement (Utile pour les tests)
      else:
        checkpoint.save(file_prefix = checkpoint_prefix)   
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / len(inputTensor)))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  
  #Création d'un objet qui va être une variable contenant toutes les informations nécessaires pour les futures prédictions
  config = {}
  config['bufferSize'] = bufferSize
  config['batchSize'] = batchSize
  config['embeddingDim'] = embeddingDim
  config['units'] = units
  config['epoch'] = epochSize
  config['vocabInputSize'] =  vocabInputSize
  config['vocabTargetSize'] = vocabTargetSize
  config['inputSentencesSize'] = inputSentencesSize
  config['targetSentencesSize'] = targetSentencesSize
  
  #Le programme enregistre le fichier de configuration dans le répertoire indiqué par l'utilisateur en JSON
  with open('{}config.json'.format(checkpointDirectory), 'w', encoding='UTF-8') as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
  #Le programme enregistre le tokenizer des 2 langues, utile pour les traductions
  with open('{}inputVocab.pickle'.format(checkpointDirectory), 'wb') as handle:
    pickle.dump(inputVocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('{}targetVocab.pickle'.format(checkpointDirectory), 'wb') as handle:
    pickle.dump(targetVocab, handle, protocol=pickle.HIGHEST_PROTOCOL)