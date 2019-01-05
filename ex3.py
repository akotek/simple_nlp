import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

TOP_TOKENS = 20


def simple_tokenize(data):
    # chopping off punctuation, save only letters,
    # lower words then tokenize:
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(data.lower())


def count_tokens(tokens, n=None):
    freq_counter = nltk.FreqDist(tokens)
    return freq_counter.most_common(n)


def plot_log(x, y):
    plt.xlabel('rank')
    plt.ylabel('frequencies')
    plt.plot(np.log(x), np.log(y))  # plot logarithm of results
    plt.show()


def q_b(tokens):
    tokens_count = count_tokens(tokens) # tokens count of form  [...(token, count)...]
    x = np.arange(0, len(tokens_count))
    y = np.array([i[1] for i in tokens_count])
    plot_log(x, y)
    print(tokens_count[:TOP_TOKENS])  # print top 20x:


def q_c(tokens):
    # remove stopwords and plot:
    stopwords_en = set(stopwords.words("english"))
    filtered_words = [word for word in tokens if word not in stopwords_en]
    q_b(filtered_words)


def q_d(tokens):
    # stem text and plot:
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(tkn) for tkn in tokens]
    q_b(stemmed_tokens)


def main():
    path = "hamlet.txt"
    with open(path, "r") as file:
        data = file.read()
        tokens = simple_tokenize(data)
        # --- (b) ---
        q_b(tokens)
        # --- (c) ---
        q_c(tokens)
        # --- (d) ---
        q_d(tokens)


if __name__ == '__main__':
    main()