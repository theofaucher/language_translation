from argparse import ArgumentParser, Namespace #argparse est utile pour exécuter le programme avec des arguments
from train import train #Fonction d'entraînement
from test import test#Fonction de test
import tensorflow as tf
import os #Interaction avec l'OS

#Permet d'exécuter le programme sur le GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' 
physical_devices = tf.config.list_physical_devices('GPU') 

def main():

    #Permet de lancer le programme avec des arguments
    parser = ArgumentParser(description='train model from data') 

    parser.add_argument('--mode', help='train or test', metavar='MODE',
                        default='train')

    #Paramètres d'apprentissage        
    parser.add_argument('--config-name', help='config json path', metavar='DIR')

    parser.add_argument('--sentences-size', help='sentences <default: 10 000>', metavar='INT',
                        type=int, default=30000)
    parser.add_argument('--batch-size', help='batch size <default: 64>', metavar='INT',
                        type=int, default=64)
    parser.add_argument('--epoch', help='epoch number <default: 10>', metavar='INT',
                        type=int, default=10)
    parser.add_argument('--embedding-dim', help='embedding dimension <default: 256>',
                        metavar='INT', type=int, default=256)
    parser.add_argument('--units', help='units <default: 1024>', metavar='INT',
                        type=int, default=1024)

    args = parser.parse_args()

    #Si l'argument mode est train, on lance un entraînement
    if args.mode == 'train':
        train(args) #Permet de lancer un entraînement
    #Sinon si c'est test, on exécute la fonction de test
    elif args.mode == 'test':
        #Si aucun argument n'est précisé, on ferme le programme
        if(args.config_name == None):
            print('Aucun fichier de configuration est indiqué dans les paramètres.')
            quit()
        test(args.config_name) #Permet de lancer un test d'une configuration particulière


if __name__ == '__main__':
    main()