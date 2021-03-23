from flask import Flask, render_template, request
import os
import json
import pickle
from argparse import ArgumentParser, Namespace
import tensorflow as tf
from model import Encoder, BahdanauAttention, Decoder
from libProject import translateAPI

parser = ArgumentParser(description='launch api with a custom training')
parser.add_argument('--config-fren', help='config json path', metavar='DIR')
parser.add_argument('--config-enfr', help='config json path', metavar='DIR')
args = parser.parse_args()

app = Flask(__name__)
checkpointAndConfigDirectoryENFR = './outputs/{}/'.format(args.config_enfr)
configTrainENFR = json.load(open(checkpointAndConfigDirectoryENFR + 'config.json', 'r', encoding='UTF-8'))
batchSize = 1 #1 setence to predict
encoderENFR = Encoder(configTrainENFR['vocabInputSize'], configTrainENFR['outputSize'], configTrainENFR['units'], batchSize)
decoderENFR = Decoder(configTrainENFR['vocabTargetSize'], configTrainENFR['outputSize'], configTrainENFR['units'], batchSize)
optimizer = tf.keras.optimizers.Adam()
checkpointENFR = tf.train.Checkpoint(optimizer=optimizer, encoder=encoderENFR, decoder=decoderENFR)
managerENFR = tf.train.CheckpointManager(checkpointENFR, checkpointAndConfigDirectoryENFR, max_to_keep=3)
checkpointENFR.restore(managerENFR.latest_checkpoint)
with open('./outputs/{}/inputVocab.pickle'.format(args.config_enfr), 'rb') as handle:
    inputVocabENFR = pickle.load(handle)
with open('./outputs/{}/targetVocab.pickle'.format(args.config_enfr), 'rb') as handle:
    targetVocabENFR = pickle.load(handle)
input_lang_tokenizerENFR = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
input_lang_tokenizerENFR.word_index = inputVocabENFR
target_lang_tokenizerENFR = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
target_lang_tokenizerENFR.word_index = targetVocabENFR

checkpointAndConfigDirectoryFREN = './outputs/{}/'.format(args.config_fren)
configTrainFREN = json.load(open(checkpointAndConfigDirectoryFREN + 'config.json', 'r', encoding='UTF-8'))
encoderFREN = Encoder(configTrainFREN['vocabInputSize'], configTrainFREN['outputSize'], configTrainFREN['units'], batchSize)
decoderFREN = Decoder(configTrainFREN['vocabTargetSize'], configTrainFREN['outputSize'], configTrainFREN['units'], batchSize)
checkpointFREN = tf.train.Checkpoint(optimizer=optimizer, encoder=encoderFREN, decoder=decoderFREN)
managerFREN = tf.train.CheckpointManager(checkpointFREN, checkpointAndConfigDirectoryFREN, max_to_keep=3)
checkpointFREN.restore(managerFREN.latest_checkpoint)
with open('./outputs/{}/inputVocab.pickle'.format(args.config_fren), 'rb') as handle:
    inputVocabFREN = pickle.load(handle)
with open('./outputs/{}/targetVocab.pickle'.format(args.config_fren), 'rb') as handle:
    targetVocabFREN = pickle.load(handle)
input_lang_tokenizerFREN = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
input_lang_tokenizerFREN.word_index = inputVocabENFR
target_lang_tokenizerFREN = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
target_lang_tokenizerFREN.word_index = targetVocabENFR


# Renvoie les paramètres reçu au client
@app.route('/translate/', methods=['GET'])
def translate():
    source = request.args.get('source')
    destination = request.args.get('destination')

    original = request.args.get('original')
    if(original == ''):
        return ''
    if(source == 'fr' and destination == 'en'):
        sentence = translateAPI(original, configTrainFREN['vocabTargetSize'], configTrainFREN['vocabInputSize'], configTrainFREN['inputSentencesSize'], configTrainFREN['targetSentencesSize'], inputVocabFREN, targetVocabFREN, configTrainFREN['units'], encoderFREN, decoderFREN)
        print(sentence)
        return sentence
    elif(source == 'en' and destination == 'fr'):
        sentence = translateAPI(original, configTrainENFR['vocabTargetSize'], configTrainENFR['vocabInputSize'], configTrainENFR['inputSentencesSize'], configTrainENFR['targetSentencesSize'], inputVocabENFR, targetVocabENFR, configTrainENFR['units'], encoderENFR, decoderENFR)
        print(sentence)
        return sentence
# Permet de renvoyer la page html au personne voulant accéder à translate.ukio.fr
@app.route('/', methods=['GET'])
def homePage():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
