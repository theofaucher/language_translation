from argparse import Namespace
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #On ne voit pas les infos, les warning, et les errors de tensorflow (Permet une meilleure visibilité dans la console)
import tensorflow as tf
import time
import pickle
import json
import random
from sklearn.model_selection import train_test_split
from libProject import loadAndCreateDataset, statsDataPreprocessed, createDataset, trainStep, translate
from model import Encoder, BahdanauAttention, Decoder

def train(args: Namespace):

  pathFile =  tf.keras.utils.get_file('fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
  extract=True)
  targetTensor, targetVocab, targetSentencesSize, inputTensor, inputVocab, inputSentencesSize = loadAndCreateDataset(os.path.dirname(pathFile)+"/fra.txt", args.sentences_size)
  print('Datasets chargées')  
  
  inputTensorTrain, inputTensorValidation, targetTensorTrain, targetTensorValidation = train_test_split(inputTensor, targetTensor, test_size=0.2)
  statsDataPreprocessed(inputTensor, targetTensor, inputVocab, targetVocab)   
  nombreEntrainement = args.epoch
  batchSize = args.batch_size
  units = args.units
  
  bufferSize = len(inputTensor)
  outputSize = args.embedding_dim    
  
  inputBatch, targetBatch, dataset = createDataset(inputTensorTrain, targetTensorTrain, batchSize, bufferSize)   
  
  vocabInputSize = len(inputVocab.word_index)+1
  vocabTargetSize = len(targetVocab.word_index)+1
  encoderModel = Encoder(vocabInputSize, outputSize, units, batchSize)
  encodeHiddenLayers = encoderModel.initialize_hidden_state()
  encodedOutputLayers, encodeHiddenLayers = encoderModel(inputBatch, encodeHiddenLayers)
  
  print ('Encoder output shape: (batch size, sequence length, units) {}'.format(encodedOutputLayers.shape))
  print ('Encoder Hidden state shape: (batch size, units) {}'.format(encodeHiddenLayers.shape))
  
  attention_layer = BahdanauAttention(10)
  attention_result, attention_weights = attention_layer(encodeHiddenLayers, encodedOutputLayers)
  
  print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
  print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
  
  decoder = Decoder(vocabTargetSize, outputSize, units, batchSize)
  sample_decoder_output, _, _ = decoder(tf.random.uniform((batchSize, 1)),encodeHiddenLayers, encodedOutputLayers)
  
  print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
  optimizer = tf.keras.optimizers.Adam()
  lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  
  if(args.config_name == None):
    configName = random.randint(0, 9999)
  else:
    configName = args.config_name
  checkpointDirectory = './outputs/{}/'.format(configName)
  checkpoint_prefix = os.path.join(checkpointDirectory, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                               encoder=encoderModel,
                               decoder=decoder)
  
  for epoch in range(nombreEntrainement):
      start = time.time()
      enc_hidden = encoderModel.initialize_hidden_state()
      total_loss = 0     
      for (batch, (inp, targ)) in enumerate(dataset.take(len(inputTensorTrain))):
        batch_loss = trainStep(inp, targ, enc_hidden, encoderModel, decoder, targetVocab, lossObject, optimizer, batchSize)
        total_loss += batch_loss      
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)      
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / len(inputTensorTrain)))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  
  config = {}
  config['bufferSize'] = bufferSize*0.8
  config['batchSize'] = batchSize
  config['outputSize'] = outputSize
  config['units'] = units
  config['epoch'] = nombreEntrainement
  config['vocabInputSize'] =  vocabInputSize
  config['vocabTargetSize'] = vocabTargetSize
  config['inputSentencesSize'] = inputSentencesSize
  config['targetSentencesSize'] = targetSentencesSize
  
  with open('{}/config.json'.format(checkpointDirectory), 'w', encoding='UTF-8') as fout:
    json.dump(config, fout, indent=2, sort_keys=True)
  with open('./outputs/{}/inputVocab.pickle'.format(configName), 'wb') as handle:
    pickle.dump(inputVocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('./outputs/{}/targetVocab.pickle'.format(configName), 'wb') as handle:
    pickle.dump(targetVocab, handle, protocol=pickle.HIGHEST_PROTOCOL)