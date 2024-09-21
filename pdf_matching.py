import string
import argparse
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text(filepath):
    """
    Extract text from a PDF file using the pypdf library and clean it up by removing redundant whitespaces.

    Parameters
    ----------
    filepath : pathlib.Path or str
        Path to the PDF file

    Returns
    -------
    str
        Text extracted from the PDF file
    """
    reader = PdfReader(filepath)
    filetext = ""
    for page in reader.pages:
        pagetext = page.extract_text()
        pagetext = pagetext.strip() # remove leading and trailing white space
        pagetext = " ".join(pagetext.split()) # replace multiple consecutive white space characters with a single space
        filetext += pagetext

    return filetext


def preprocess_text(text):
    """
    Preprocess the text by converting it to lowercase, tokenizing it, removing stopwords, punctuation, and stemming the words.

    Parameters
    ----------
    text : str
        Text to be preprocessed

    Returns
    -------
    str
        Preprocessed text    
    """
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens \
              if token.isalpha() and \
              token not in string.punctuation and \
              token not in stopwords.words('english')]
    
    return ' '.join(tokens)


def process_files(files):
    """
    Process a list of PDF files by extracting text and preprocessing it.

    Parameters
    ----------
    files : list of pathlib.Path
        List of paths to the PDF files

    Returns
    -------
    list of str, list of str
        List of filenames and list of preprocessed texts
    """
    filenames = []
    texts = []
    for filepath in files:
        filenames.append(filepath.stem)
        text = extract_text(filepath)
        text = preprocess_text(text)
        texts.append(text)

    return filenames, texts


def calculate_pairwise_similarity_matrix(texts):
    """
    Calculate the pairwise similarity matrix between a list of texts using the TF-IDF vectorizer and cosine similarity.

    Parameters
    ----------
    texts : list of str
        List of preprocessed texts

    Returns
    -------
    numpy.ndarray
        Pairwise similarity matrix
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf)

    return similarity_matrix


def find_matching_pairs(similarity_matrix, threshold):
    """
    Find matching pairs of texts based on the similarity matrix and a threshold value.

    Parameters
    ----------
    similarity_matrix : numpy.ndarray
        Pairwise similarity matrix
    threshold : float
        Threshold value for similarity

    Returns
    -------
    list of tuple, list of float
        List of matching pairs of indices and list of similarity values
    """
    matching_pairs = np.argwhere(np.triu(similarity_matrix, k=1) > threshold)
    similarity_values = similarity_matrix[matching_pairs[:,0], matching_pairs[:,1]]

    sort_indices = np.argsort(similarity_values)[::-1]
    matching_pairs = matching_pairs[sort_indices]
    similarity_values = similarity_values[sort_indices]

    return matching_pairs.tolist(), similarity_values.tolist()


def display_matches(matches, similarities, filenames):
    """
    Display the matching pairs of filenames and their similarity values.

    Parameters
    ----------
    matches : list of tuple
        List of matching pairs of indices
    similarities : list of float
        List of similarity values
    filenames : list of str
        List of filenames

    Returns
    -------
    None
    """
    for i, ((matchx, matchy), similarity) in enumerate(zip(matches, similarities)):
        print('Match {}: '.format(i+1))
        print('\tFile 1: {}'.format(filenames[matchx]))
        print('\tFile 2: {}'.format(filenames[matchy]))
        print('\tSimilarity: {:.2f}'.format(similarity))


def main(folder, threshold):
    """
    Main function to find matching pairs of PDF files in a folder based on their text similarity.

    Parameters
    ----------
    folder : str
        Path to the folder containing PDF files
    threshold : float
        Threshold value for similarity

    Returns
    -------
    None
    """
    folder = Path(folder)
    files = [x for x in folder.glob('*.pdf')]

    filenames, texts = process_files(files)

    similarity_matrix = calculate_pairwise_similarity_matrix(texts)

    matching_pairs, similarities = find_matching_pairs(similarity_matrix, threshold)
    display_matches(matching_pairs, similarities, filenames)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PDF File Matcher')
    parser.add_argument('--dir', help='Folder', type=str)
    parser.add_argument('--threshold', help='Threshold', type=float, default=0.7)
    args = parser.parse_args()

    main(folder=args.dir, threshold=args.threshold)