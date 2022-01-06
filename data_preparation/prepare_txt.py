# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:30:58 2022

@author: Sagun Shakya

Simple utility to prepare a txt file for input to the Gensim Word Vector model.
"""

import pandas as pd
from datetime import datetime

# Local module.
from utils import *

#%% Filename: "D:/ML_projects/Word Vectors/sample_data/Merged_series_2.pkl"
"""
Data description:
-----------------
count    60809.00
mean        12.88
std          8.30
min          1.00
25%          7.00
50%         11.00
75%         16.00
max         89.00
"""

filename1 = r"D:/ML_projects/Word Vectors/sample_data/Merged_series_2.pkl"
df1 = pd.read_pickle(filename1)
df1 = df1.str.join(" ")
convert_to_txt(series = df1, output_dir = 'data_preparation/input_files', filename = 'Merged_series_2.txt')

#%% Filename: E:\Study Materials for project\Datasets\Setopati

root_dir = r'E:\Study Materials for project\Datasets\Setopati'

# Collect the names of all the .txt files from the root directory and its sub-directories.
filenames = get_filenames(root_dir)

# Make a list of series for each text file and concatenating them to create a big corpus.
series_list = [prepare_series(filename) for filename in filenames]

# Concatenating the list of series.
setopati_corpus = pd.concat(series_list, axis = 0, ignore_index = True).reset_index(drop = True)
setopati_corpus.drop([621], inplace=True)

# Prepare text file.
convert_to_txt(series = setopati_corpus, output_dir = 'data_preparation/input_files/txt', filename = 'setopati_all.txt')  

#%% Concatenate both Merged_series_2.txt and setopati_all.txt.
root_dir = 'input_files/txt'
filenames = [root_dir + '/' + name for name in os.listdir(root_dir)]

print('Number of files: ', len(filenames))

# Current date.
date = datetime.today().strftime('%Y-%m-%d')

# Output filename.
out_filename = root_dir + '/' + 'Merged_series_2__setopati_all_' + date + ".txt"

# Concatenate the contents.
concat_text_files(path_filenames = filenames, outputfile_path = out_filename)
