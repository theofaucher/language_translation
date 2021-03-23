from argparse import ArgumentParser, Namespace
from train import train 
from test import test
import tensorflow as tf
import os

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #Permet d'éxecuté le programme sur le GPU

#physical_devices = tf.config.list_physical_devices('GPU') 
#/bin/bash: q: command not found

def main():

    parser = ArgumentParser(description='train model from data')

    parser.add_argument('--mode', help='train or test', metavar='MODE',
                        default='train')
        
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

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == '__main__':
    main()
