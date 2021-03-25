#Librairie utile dans ce programme
import json, pickle, os #Interaction avec l'OS
from argparse import ArgumentParser, Namespace #Permet d'avoir des arguments à l'exécution du programme
import tensorflow as tf
from model import Encoder, BahdanauAttention, Decoder #Import du modèle de l'IA depuis le fichier model
from libProject import translate, createNetworkAndApplyCheckPointsAndConfig #Import de ma librairie

def test(configName):
    '''
    Programme CLI permettant d'essayer un entraînement

    configName: Nom du répertoire qui contient l'entraînement que vous voulez essayer (ex : test1)
    '''

    #Création du réseau avec les paramètres présent dans le répertoire fourni en paramètre
    encoder, decoder, inputVocab, targetVocab, configTrain = createNetworkAndApplyCheckPointsAndConfig(configName)

    #Tant que vrai
    while True:
        sentence = input('Ecrivez une phrase: ') #Le programme récupère la phrase saisie après appui de la touche entrée

        #Si la phrase est vide, on quitte le programme
        if sentence == '':
            break

        #Appel de la fonction translate pour effectuer une traduction
        translate(sentence, configTrain['vocabTargetSize'], configTrain['vocabInputSize'], configTrain['inputSentencesSize'], configTrain['targetSentencesSize'], inputVocab, targetVocab, configTrain['units'], encoder, decoder)


def main():
    pass

if __name__ == '__main__':
    main()