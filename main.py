# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:00:33 2021

@author: Sagun Shakya
"""

# Importing the necessary libraries.
import pandas as pd
import multiprocessing
import argparse
from time import time

# Local Modules.
from model.word2vec import Word2Vec_model

# Filter Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Count the number of cores in the CPU.
cores = multiprocessing.cpu_count()
print("Number of CPU cores: ", cores)
print()

def train(args):
    '''
    This function will be called by the main function.
    '''
    
    # If the input is a txt file, then it is passed as it is.
    # If it is a pickle file, it will be parsed into a list and then fed into the model for training.
    if not args.txt_file:
        file = pd.read_pickle(args.file_path)
        document = file.tolist()
    else:
        document = args.file_path
        
    # "document" contains the input to the model to start the self supervised training.
    
    # Initialize the model object.
    w2v = Word2Vec_model(sg = args.sg,
                         sentence_istxt = args.txt_file,
                         loss = args.loss,
                         min_count = args.min_count,
                         window = args.window,
                         size = args.size,
                         sample = args.sample,
                         alpha = args.alpha,
                         min_alpha = args.min_alpha,
                         negative = args.negative,
                         workers = args.workers,
                         num_iterations = args.num_iterations,
                         random_state = args.random_state)
    
    print('Training started.')
    print('...')
    start = time()
    
    # Train the model.
    model = w2v.train_model(document)
    
    end = time()
    print(f'Training completed.\nTime Elapsed : {round(end - start, 5)} seconds.')
    
    # Save the model if save_model == True.
    if args.save_model:
        w2v.save_model(args.model_dir, model)
    
    # Save the Keyed Vectors if save_keyedvectors == True.
    if args.save_keyedvectors:
        w2v.save_keyed_vectors(args.keyedvector_dir, model)
        


def main():
    '''
    Takes in argument for training the models.
    '''
    
    # Default directories to save the model and keyed vectors.
    output_model_dir = 'output_dir/model_dir'
    output_keyed_vector_dir = 'output_dir/keyedvectors_dir'
    
    # Parse the arguments from the user.
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', action = 'store_true', required = True, help = 'Is the input corpus a txt file?')
    parser.add_argument('file_path', type = str, help = 'Path to the corpus file.')
    parser.add_argument('--save_model', action = 'store_true', help = 'Whether to store the model file.')
    parser.add_argument('--save_keyedvectors', action = 'store_true', help = 'Whether to store the keyed vectors.')
    parser.add_argument('--model_dir', type = str, default = output_model_dir, 
                        help = 'Path to the directory that holds the saved model.')
    parser.add_argument('--keyedvector_dir', type = str, default = output_keyed_vector_dir, 
                        help = 'Path to the directory that holds the saved Keyed Vectors.')

    parser.add_argument('--sg', type = int, default = 1, help = 'Whether to use skipgram (or C-BOW).')
    parser.add_argument('-n', '--size', type = int, default = 100, help = 'Embedding dimension.')
    parser.add_argument('-w', '--window', type = int, default = 5, help = 'Context size.')
    parser.add_argument('-a', '--alpha', type = float, default = 0.05, help = 'Learning Rate.')
    parser.add_argument('--min_alpha', type = float, default = 0.0001, 
                        help = 'Learning rate will linearly drop to min_alpha as training progresses.')
    parser.add_argument('--loss', type = int, default = 0, help = 'Whether to use heirarchical loss.')
    parser.add_argument('--min_count', type = int, default = 3, help = 'Minimum frequency of a token to qualify for training.')
    parser.add_argument('--sample', type = float, default = 6e-5, help = 'Set threshold for occurrence of words.')
    parser.add_argument('--negative', type = int, default = 10, help = 'How many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.')
    parser.add_argument('--workers', type = int, default = int(cores) - 1, help = 'Number of CPU cores.')
    parser.add_argument('-e', '--num_iterations', type = int, default = 5, help = 'Number of epochs.')
    parser.add_argument('--random_state', type = int, default = 100, help = 'Seed for reproducibility.')
    
    args = parser.parse_args()
    train(args)
    
if __name__ == '__main__':
    main()