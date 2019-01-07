import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import PorterStemmer
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

TOP_TOKENS = 20


def simple_tokenize(data):
    # chopping off punctuation, save only letters,
    # lower words then tokenize:
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(data)


def count_tokens(tokens, n=None):
    tokens = [t.lower() for t in tokens]
    freq_counter = nltk.FreqDist(tokens)
    return freq_counter.most_common(n)


def plot_log(x, y, title):
    plt.xlabel('rank')
    plt.ylabel('frequencies')
    plt.plot(np.log(x + 1), np.log(y))  # plot logarithm of results
    plt.title("Plot for question " + title + '.')
    plt.show()


def q_b(tokens, question_letter):
    tokens_count = count_tokens(tokens) # tokens count of form  [...(token, count)...]
    x = np.arange(0, len(tokens_count))
    y = np.array([i[1] for i in tokens_count])
    plot_log(x, y, question_letter)
    print('List for question ' + question_letter + " :")
    print(tokens_count[:TOP_TOKENS])  # print top 20x:


def q_c(tokens):
    # remove stopwords and plot:
    stopwords_en = set(stopwords.words("english"))
    filtered_words = [word for word in tokens if word not in stopwords_en]
    q_b(filtered_words, 'c')


def q_d(tokens):
    # stem text and plot:
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(tkn) for tkn in tokens]
    q_b(stemmed_tokens, 'd')


def q_e(book_text):
    token_sentences = sent_tokenize(book_text)      # break into sentences
    token_list = []
    for sentence in token_sentences:
        words = simple_tokenize(sentence)           # leave only English letters
        tagged = nltk.pos_tag(words)                # POS tagging
        chunkGram = r"""Chunk: {<JJ.?>*<NN.?.?>+}"""        # Get only groups of nouns preceded by adjectives
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        for subtree in chunked.subtrees():                  # Group together every element that matches the pattern
            if subtree.label() == 'Chunk':
                token = ' '.join(leaf[0] for leaf in subtree.leaves())
                token_list.append(token)
    tokens_count = count_tokens(token_list)
    x = np.arange(0, len(tokens_count))
    y = np.array([i[1] for i in tokens_count])
    plot_log(x, y, 'e')
    print('List for question e:')
    print(tokens_count[:TOP_TOKENS])


def q_f():
    # Example for wrong POS tagging, NLTK POS tagging algorithm identifies 'speak' as NN in this sentance.
    sentence = 'Therefore I have entreated him along,\nWith us to watch the minutes of this night,\nThat, if again' \
               ' this apparition come,\nHe may approve our eyes and speak to it.'
    words = simple_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    print('Regarding question f:')
    print('The tagging of the sentence by NLYK POS tagging algorithm on the following sentence:')
    print('"' + sentence + '"')
    print(tagged)


def main():
    path = "hamlet.txt"
    with open(path, "r") as file:
        data = file.read()
    tokens = simple_tokenize(data)
    # --- (b) ---
    q_b(tokens, 'b')
    # --- (c) ---
    q_c(tokens)
    # --- (d) ---
    q_d(tokens)
    # --- (e) ---
    q_e(data)
    # --- (f) ---
    q_f()


if __name__ == '__main__':
    main()
