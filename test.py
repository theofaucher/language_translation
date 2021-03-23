import json, pickle, os
from argparse import ArgumentParser, Namespace
import tensorflow as tf
from model import Encoder, BahdanauAttention, Decoder
from libProject import  preprocessSentences, translate

def test(args: Namespace):

    if(args.config_name == None):
        print('Aucun fichier de configuration est indiqué dans les paramètres.')
        quit()

    checkpointAndConfigDirectory = './outputs/{}/'.format(args.config_name)
    configTrain = json.load(open(checkpointAndConfigDirectory + 'config.json', 'r', encoding='UTF-8'))
    batchSize = 1 #1 setence to predict
    encoder = Encoder(configTrain['vocabInputSize'], configTrain['outputSize'], configTrain['units'], batchSize)
    decoder = Decoder(configTrain['vocabTargetSize'], configTrain['outputSize'], configTrain['units'], batchSize)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(checkpoint, checkpointAndConfigDirectory, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    vocabsDirectory = './outputs/{}/'.format(args.config_name) 

    with open( vocabsDirectory + 'inputVocab.pickle', 'rb') as handle:
        inputVocab = pickle.load(handle)

    with open(vocabsDirectory + 'targetVocab.pickle', 'rb') as handle:
        targetVocab = pickle.load(handle)

    while True:
        sentence = input(
            'Ecrivez une phrase: ')

        if sentence == '':
            break
        
        input_lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        input_lang_tokenizer.word_index = inputVocab

        target_lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>') # , oov_token='<unk>'
        target_lang_tokenizer.word_index = targetVocab

        translate(sentence, configTrain['vocabTargetSize'], configTrain['vocabInputSize'], configTrain['inputSentencesSize'], configTrain['targetSentencesSize'], inputVocab, targetVocab, configTrain['units'], encoder, decoder)




def main():
    pass


if __name__ == '__main__':
    main()
