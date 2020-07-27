import sys
import numpy as np
import pandas as pd
import re
from collections import defaultdict


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
            for word in get_words(text):
                # Increment its count by 1
                bow[word][ind] += 1
                cat_word_counts[ind] += 1

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
    cats_info = [bow, np.log(p)]

    return cats_info


# Predict
def predict(reviews, cats_info):
    predictions = np.zeros(reviews.shape[0])

    no_of_cat = cats_info[1].size

    # For each review
    for rev_index, rev in enumerate(reviews):

        # Likelihood Probability
        likelihood_prob = np.zeros(no_of_cat)

        for word in get_words(rev):
            likelihood_prob += cats_info[0][word]

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
