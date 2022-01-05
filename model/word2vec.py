# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:30:37 2022

@author: Sagun Shakya
"""
import os
import errno
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class Word2Vec_model:
    '''
        Word2Vec class.

        Training can be done in two of the following modes:
            - Skipgram (Code : 1).
            - CBOW (Code : 0).
        In training mode, the input file must be provided in either of the two formats:
            - Text file.
                Here, each line is a sentence and its tokens are separated by a whitespace.
                To use this, flag the parameter 'sentence_istext' as True. It is True by default.
            - Iterable.
                To use this, one must tokenize the sentence and place all such sentence groups in a list.
                Flag the parameter 'sentence_istext' as False.

        Gensim official documentation: https://radimrehurek.com/gensim/models/word2vec.html
  
        Parameters:
            sg -- (1 or 0) Whether the training is done under skipgram (or C-BOW). Default = 1 (skipgram).
            sentence_istxt -- Bool. If the sentence to be provided is a txt file. Default = True 
                              (IF False, the sentences must be tokenized and placed in a list.)
            loss -- (1 or 0) Loss function. 0 -> Negative Sampling & 1 -> Heirarchical Loss. Default = 0.
            min_count -- Minimum frequency of a token to be considered.
            window -- Context size. Default = 3 -> 3 tokens will be collected from the left and the right when trainng.
            size -- int. The dimension of the word vectors in the output.
            sample -- Set threshold for occurrence of words. 
                      Those that appear with higher frequency in the training data will be randomly down-sampled; 
                      default is 1e-3, useful range is (0, 1e-5).
            alpha -- Learning Rate. Default = 0.05.
            min_alpha -- Learning rate will linearly drop to min_alpha as training progresses.
            negative --  How many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
            workers -- Number of CPU cores to employ.
            num_iterations -- Number of epochs to run the training loop. Default = 5.
            random_state -- For reproducibility.

    '''
       
    def __init__(self,
                 sg = 1,
                 sentence_istxt = True,
                 loss = 0,
                 min_count = 2,
                 window = 3,
                 size = 300,
                 sample = 6e-5, 
                 alpha = 0.05, 
                 min_alpha = 0.0007, 
                 negative = 10,
                 workers = 4,
                 num_iterations = 5,
                 random_state = 100):
        
         
        self.sg = sg
        self.sentence_istxt = sentence_istxt
        self.loss = loss
        self.min_count=min_count
        self.window = window
        self.size = size
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.workers = workers
        self.num_iterations = num_iterations
        self.random_state = random_state
        
    def train_model(self, file_path):
        '''
        Returns a Word2Vec model object.
        Can perform further training using .train() method. (See Documentation.)
        
        Parameters:
            file_path -- When sentence_istxt == True, this is path to the corpus file.
                         When sentence_istxt == False, this is the iterable.
        '''    
        
        if self.sentence_istxt: # For .txt File.
        
            # Preparing the input sentences for training.    
            if not os.path.exists(file_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
            else:
                sentence = LineSentence(file_path)
        
        else:   # For Iterable.
            sentence = file_path

        # Training the Word2Vec model.
        model = Word2Vec(sentence,
                         sg = self.sg,
                         min_count = self.min_count,
                         window = self.window,
                         size = self.size,
                         sample = self.sample,
                         alpha = self.alpha,
                         hs = self.loss,
                         min_alpha = self.min_alpha,
                         negative = self.negative,
                         workers = self.workers,
                         iter = self.num_iterations,
                         seed = self.random_state)

        return model
        

    def save_model(self, output_dir, model):
        '''
        Saves the model as a whole.
        
        To load such saved file ("mymodel.bin"), use:
            from gensim.models import Word2Vec
            xx = Word2Vec.load("mymodel.bin")
        
        Parameters:
            output_dir -- Path to the directory to save the file.
            model -- Trained Word2Vec model object.
            
        Note:
            If there is no directory as passed in the argument 'output_dir',
            a new directory will be created as 'output_dir/model_dir' and 
            the file will be saved there.

        '''
        dimensions = str(self.size)
        mode = 'skipgram' if (self.sg == 1) else 'cbow'
        
        # Set filename.
        output_filename = 'model_word2vec_{}d_{}.bin'.format(dimensions, mode)

        # Check to see if the specified directory exists.
        # If it doesn't exist, make a new directory named "output_dir/model_dir".
        if not os.path.exists(output_dir):
            print(f'Directory {output_dir} does not exist in this project.\nMaking new directory named output_dir/model_dir...\n')
            os.makedirs('output_dir/model_dir', exist_ok = True)
            output_dir = os.path.join('output_dir', 'model_dir')
            
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Save model.
        model.save(output_filepath)
 
        
    def save_keyed_vectors(self, output_dir, model):
        '''
        Saves the keyed vector.
        
        To load such saved file ("lolol.wordvectors"), use:
            from gensim.models import KeyedVectors
            xx = KeyedVectors.load("lolol.wordvectors", mmap='r')
        
        Parameters:
            output_dir -- Path to the directory to save the file.
            model -- Trained Word2Vec model object.

        Note:
            If there is no directory as passed in the argument 'output_dir',
            a new directory will be created as 'output_dir/keyedvectors_dir' and 
            the file will be saved there.

        '''
  
        dimensions = str(self.size)
        mode = 'skipgram' if (self.sg == 1) else 'cbow'
        
        # Set filename.
        output_filename = 'model_word2vec_{}d_{}.wordvectors'.format(dimensions, mode)

        # Check to see if the specified directory exists.
        # If it doesn't exist, make a new directory named "output_dir/model_dir".
        if not os.path.exists(output_dir):
            print(f'Directory {output_dir} does not exist in this project.\nMaking new directory named output_dir/keyedvectors_dir...\n')
            os.makedirs('output_dir/keyedvectors_dir', exist_ok = True)
            output_dir = os.path.join('output_dir', 'keyedvectors_dir')
            
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Save model.
        model.wv.save(output_filepath)
        
        
        
        







