# Word2Vec
Word2Vec was introduced in two papers between September and October 2013, by a team of researchers at Google. Along with the papers, the researchers published their implementation in C. The Python implementation was done soon after the 1st paper, by Gensim.

### The need
The goal with word2vec is to translate text into an n-dimensional vector so that they can then be processed using operations from linear algebra.

### Description
Word2vec is actually a collection of two different methods: continuous bag-of-words (CBOW) and skip-gram.
- Given a word in a sentence, lets call it w(t) (also called the center word or target word), **CBOW** uses the context or surrounding words as input. For instance, if the context window C is set to C=5, then the input would be words at positions w(t-2), w(t-1), w(t+1), and w(t+2). Basically the two words before and after the center word w(t). Given this information, CBOW then tries to predict the target word.
- The second method, **skip-gram** is the exact opposite. Instead of inputting the context words and predicting the center word, we feed in the center word and predict the context words. This means that w(t) becomes the input while w(t-2), w(t-1), w(t+1), and w(t+2) are the ideal output.
- Because we can't send text data directly through a matrix, we need to employ **one-hot encoding**. This means we have a vector of length **v** where **v** is the total number of unique words in the text corpus.

### Parameters

    min_count = int - Ignores all words with total absolute frequency lower than this - (2, 100)

    window = int - The maximum distance between the current and predicted word within a sentence. E.g. window words on the left and window words on the left of our target - (2, 10)

    size = int - Dimensionality of the feature vectors. - (50, 300)

    sample = float - The threshold for configuring which higher-frequency words are randomly downsampled. Highly influencial. - (0, 1e-5)

    alpha = float - The initial learning rate - (0.01, 0.05)

    min_alpha = float - Learning rate will linearly drop to min_alpha as training progresses. To set it: alpha - (min_alpha * epochs) ~ 0.00

    negative = int - If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drown. If set to 0, no negative sampling is used. - (5, 20)

    workers = int - Use these many worker threads to train the model (=faster training with multicore machines)


### Training of the model:

#### Parameters of the training:
    total_examples = int - Count of sentences;
    epochs = int - Number of iterations (epochs) over the corpus - [10, 20, 30]

    
### 2-Dimensional visualization using t-SNE.
Our goal in this section is to plot our 300 dimensions vectors into 2 dimensional graphs, and see if we can spot interesting patterns.
To make the visualizations more relevant, we will look at the relationships between a query word (in `red`), its most similar words in the model (in **blue**), and other words from the vocabulary (in **green**).

```sh
def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=50).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
```





