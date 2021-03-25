from flask import Flask, render_template, request #Permet la création et la gestion d'un service web
import os, json, pickle #Interaction avec l'OS
from argparse import ArgumentParser, Namespace #Permet d'avoir des arguments à l'exécution du programme
import tensorflow as tf
from model import Encoder, BahdanauAttention, Decoder #Modèle de l'IA
from libProject import createNetworkAndApplyCheckPointsAndConfig, translateAPI #Fonction nécessaire pour créer l'IA et traduire du contenu

parser = ArgumentParser(description='launch api with a custom training')
parser.add_argument('--config-fren', help='config json path', metavar='DIR') #Argument pour de la traduction fr -> en 
parser.add_argument('--config-enfr', help='config json path', metavar='DIR') #Argument pour de la traduction en -> fr
args = parser.parse_args()

app = Flask(__name__) #Création du service web

#Création du réseau avec les paramètres présent dans le répertoire fourni en paramètre
encoderENFR, decoderENFR, inputVocabENFR, targetVocabENFR, configTrainENFR = createNetworkAndApplyCheckPointsAndConfig(args.config_enfr)
encoderFREN, decoderFREN, inputVocabFREN, targetVocabFREN, configTrainFREN = createNetworkAndApplyCheckPointsAndConfig(args.config_fren)

# Requête de type get sur <ip>/translate/
@app.route('/translate/', methods=['GET'])
def translate():
    source = request.args.get('source') #Récupération de l'argument source  (Correspond à la langue d'entrée) dans la requête
    destination = request.args.get('destination') #Récupération de l'argument destination (Correspond à la langue de sortie) dans la requête
    original = request.args.get('original') #Récupération de l'argument original (qui contient le texte à traduire) dans la requête

    if(original == ''): #Si la phrase est vide, le programme revoie une chaîne de caractères vide
        return ''
    if(source == 'fr' and destination == 'en'): #Traduction de Français vers Anglais
        sentence = translateAPI(original, configTrainFREN['vocabTargetSize'], configTrainFREN['vocabInputSize'], configTrainFREN['inputSentencesSize'], configTrainFREN['targetSentencesSize'], inputVocabFREN, targetVocabFREN, configTrainFREN['units'], encoderFREN, decoderFREN)
        print(sentence)
        return sentence
    elif(source == 'en' and destination == 'fr'): #Traduction de Anglais vers Français
        sentence = translateAPI(original, configTrainENFR['vocabTargetSize'], configTrainENFR['vocabInputSize'], configTrainENFR['inputSentencesSize'], configTrainENFR['targetSentencesSize'], inputVocabENFR, targetVocabENFR, configTrainENFR['units'], encoderENFR, decoderENFR)
        print(sentence)
        return sentence

#Permet de renvoyer la page html aux personnes voulant accéder à translate.ukio.fr
@app.route('/', methods=['GET'])
def homePage():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()