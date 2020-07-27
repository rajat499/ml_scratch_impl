import sys
import numpy as np
import pandas as pd
import re
from collections import defaultdict

stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


# Return words from the Text
def get_words(text):
    # Type cast to String
    text = str(text)

    # Remove any trailing and leading whitespaces, also convert to lowercase
    text = text.strip().lower()

    # Text is split upon every char except a-z, 0-9 and '
    return re.split("[^a-z0-9']+", text)


# Read Training Data
def read_tr(file):
    reviews, labels = np.split(pd.read_csv(file).values, 2, axis=1)

    # Separate positive and negative reviews
    pos = reviews[labels == 'positive']
    neg = reviews[labels == 'negative']

    return neg, pos


# Create an array for Bag of Words
def bag_of_words(*data):
    no_of_cat = len(data)

    # A default dict for each class
    bow = defaultdict(lambda: np.ones(no_of_cat))

    # Number of words in each category
    cat_word_counts = np.zeros(no_of_cat)

    # For each cat
    for ind, cat in enumerate(data):
        # For each text
        for text in cat:
            # For each word
            word_ind = 0
            words = get_words(text)
            
            while words[word_ind] in stop_words:
                word_ind += 1
                
            while word_ind < len(words):
                # Increment its count by 1
                bow[words[word_ind]][ind] += 1
                cat_word_counts[ind] += 1
            
                big_end = word_ind + 1
                
                if big_end == len(words):
                    break
                
                while words[big_end] in stop_words:
                    big_end += 1
                    if big_end == len(words):
                        break
                
                if big_end == len(words):
                    break
                
                big = words[word_ind] + ' ' + words[big_end]
                bow[big][ind] += 1
                cat_word_counts[ind] += 1
                    
                word_ind = big_end
    
    # Add number of words with -1 as key
    bow[-1] = cat_word_counts

    return bow


# Train Naive Bayes Model
def train_nb(*data):
    no_of_cat = len(data)

    # Bag of Words
    bow = bag_of_words(*data)

    # Calculating total counts of all the words of each category
    cat_word_counts = bow[-1]

    # Delete extra entry which was keeping word count
    bow.pop(-1)

    # All words from all categories
    vocab = set(bow.keys())
    vocab_len = len(vocab)

    # Prior Probability of each category
    p = np.array([cat.size for index, cat in enumerate(data)])
    p = p / np.sum(p)

    # Computing denominator value for each category
    den = cat_word_counts + vocab_len

    # Update bow to store log likelihood probabilities
    for key, value in bow.items():
        bow[key] = np.log(value / den)

    bow.default_factory = lambda: -np.log(den)

    # Store in one place
    cats_info = [bow, np.log(p), vocab]

    return cats_info


# Predict
def predict(reviews, cats_info):
    predictions = np.zeros(reviews.shape[0])

    no_of_cat = cats_info[1].size

    # For each review
    for rev_index, rev in enumerate(reviews):

        # Likelihood Probability
        likelihood_prob = np.zeros(no_of_cat)
        
        word_ind = 0
        words = get_words(rev)
        
        while words[word_ind] in stop_words:
            word_ind += 1
            
        while word_ind < len(words):
            likelihood_prob += cats_info[0][words[word_ind]]
        
            big_end = word_ind + 1
            if big_end == len(words):
                    break
            while words[big_end] in stop_words:
                big_end += 1
                if big_end == len(words):
                    break
            
            if big_end == len(words):
                break
            
            big = words[word_ind] + ' ' + words[big_end]
            likelihood_prob += cats_info[0][big]
                
            word_ind = big_end

        # We have likelihood estimate but we need posterior probability
        post_prob = likelihood_prob + cats_info[1]

        predictions[rev_index] = np.argmax(post_prob)

    return predictions


# Main Interface
def main():
    # Command Line Arguments
    train_file, test_file, out_file = sys.argv[1:]

    # Read Training Data
    neg, pos = read_tr(train_file)

    # Train Naive Bayes Model
    cats_info = train_nb(neg, pos)

    # Read Test Data
    reviews = pd.read_csv(test_file).values

    # Predict
    predictions = predict(reviews, cats_info)

    # Write results in a file
    predictions.tofile(out_file, '\n')


# Execute if used as a script
if __name__ == '__main__':
    main()
