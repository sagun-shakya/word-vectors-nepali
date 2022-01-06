# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 22:50:52 2022

@author: Sagun Shakya

Helper functions to prepare text input for the Gensim Word Vector model.
The functions are called in "data_preparation/prepare_txt.py".

"""

# Importing the necessary libraries.
import pandas as pd
import os
import shutil
from re import split
from string import punctuation, ascii_uppercase, ascii_lowercase

# The helper functions are used in the script prepare_txt.py.

def read_text_file(filename):
    '''
    Reads the text present in the .txt file and returns the output in string format.
    
    Parameters:
    filename -- Name of the text file. e.g '1217589.txt'.
    
    Output:
    String format text.
    
    '''
    with open(file = filename, encoding = 'utf-8') as ff:
        text = ff.readlines()
        text = ' '.join(text)
        text = text.replace(u'\xa0', ' ').encode('utf-8').decode()    # Removing the \xa0 character.      
        text = text.replace('…', ' ')
        text = text.replace('“', ' ')
        text = text.replace('”', ' ')
        text = text.replace('–', ' ')
        text = text.replace('\n', '')                                 # Removing the \n character.
        ff.close()
    return text

def cleaner(sentence): 
    '''
    Simple text cleaner function.
    Removes punctuations, english letters, english digits, nepali digits 
    and other known punctuations except ।?|.
    '''
        
    nepali_digits = ''.join([chr(2406 + ii) for ii in range(10)])
    english_digits = ''.join([chr(48 + ii) for ii in range(10)])
    english_alphabets = ascii_uppercase + ascii_lowercase
    other_punctuations = '‘’' + '"#$%&\'()*+,./:;<=>@[\\]^_`{}~…“”–' + chr(8211)
    temp = nepali_digits + english_digits + english_alphabets + other_punctuations + chr(8226)
    temp = set(temp)
    
    punct = set(punctuation).difference(set("।?|!"))
    
    to_remove = ''.join(set.union(temp, punct))      
    result = sentence.translate(str.maketrans('', '', to_remove))

    return result

def generate_sentence_tokens(paragraph):
    '''
    Takes in the cleaned text. --> Splits it using delimiters. --> Converts to Series.
    '''
    temp = split('[।l?|]', paragraph)
    temp = [SENT.strip() for SENT in temp if len(SENT.strip()) > 0]
    temp = pd.Series(temp)
    return temp

def prepare_series(filename):
    '''
    Pipeline to take in the raw text file, clean it and convert it into Series.

    Parameters
    ----------
    filename : str
        Name of the text file.

    Returns
    -------
    result : Pandas Series.
        Series containing one sentence per row.

    '''
    # Read the text file.
    sample = read_text_file(filename)
    
    # Apply cleaner function.
    cleaned = cleaner(sample)
    
    # Convert to Pandas Series.
    series = generate_sentence_tokens(cleaned)
    
    # Remove rows with less than 3 words.
    lengths = series.str.split().apply(lambda x: len(x))
    
    # Ids of such instances.
    idx = lengths[lengths < 3].index
    
    # Removing such instances and resetting the index column.
    series = series.drop(idx).reset_index(drop = True)
    
    return series
    
def convert_to_txt(series, output_dir, filename = "sentences.txt"):
    '''
    Converts the Pandas Series of Strings to a txt file.
    This will be used as an input to the Gensim word vector model.
    Each instance is present in a line and its tokens are separated by a whitespace.

    Parameters
    ----------
    series : Pandas Series.
        Series containing the strings separated by a whitespace.
    output_dir : str.
        Path to the directory that will hold the output file.
    filename : str.
        Name of the output file.

    Returns
    -------
    Text file in a desired location.

    '''
    # Check to see if the specified directory exists.
    # If it doesn't exist, make a new directory named "input_files/txt".
    if not os.path.exists(output_dir):
        print(f'Directory {output_dir} does not exist in this project.')
        print('Making new directory named input_files/txt...\n')
        os.makedirs('input_files/txt', exist_ok = True)
        output_dir = os.path.join('input_files', 'txt')

    
    # Set filename.
    out_filename = os.path.join(output_dir, filename)
    
    try:
        # Convert to txt.
        num_instances = len(series)
        series.to_csv(out_filename, header = None, index = None, encoding = 'utf-8')
        return f"Converted {num_instances} instances to text file.\nStored in {output_dir}"
    except:
        return "Something went wrong."


def get_filenames(root_dir):
    '''
    Gets the filenames (.txt) from a root directory recursively.

    Parameters
    ----------
    root_dir : str
        Root Directory.

    Returns
    -------
    List. Names of all the text files.

    '''
    
    filenames = []
    for root, dirs, files in os.walk(root_dir):
    	for file in files:
    		if(file.endswith(".txt")):
    			filenames.append(os.path.join(root,file))   
    
    print(f"{len(filenames)} documents collected.")
    return filenames

def concat_text_files(path_filenames, outputfile_path):
    '''
    Concatenates the contents of two or more text files.

    Parameters
    ----------
    path_filenames : list.
        list of paths to the text files.
    outputfile_path : str
        path to the output file.

    Returns
    -------
    New text file that contains the concatenation of the input text files.

    '''

    with open(outputfile_path,'wb') as wfd:
        for f in path_filenames:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
