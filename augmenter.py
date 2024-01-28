from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import random
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def get_synonyms(word, pos=None, custom_synonyms=None):
    """
    Get synonyms for a given word.

    Parameters:
    - word (str): The input word for which synonyms are to be retrieved.
    - pos (str, optional): The part-of-speech tag. If provided, only synonyms of
      the specified part-of-speech will be considered (e.g., 'n' for nouns, 'v' for verbs).
    - custom_synonyms (dict, optional): A dictionary containing custom synonyms for words.
      Keys are words, and values are lists of custom synonyms.

    Returns:
    - List[str]: A list of synonyms for the input word.
    """

    synonyms = set()

    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())

    if custom_synonyms:
        synonyms.update(custom_synonyms.get(word, []))

    synonyms.discard(word)

    return list(synonyms)


def lexical_substitution(text, pos=None, custom_synonyms=None):
    """
    Perform lexical substitution on a given text.

    Parameters:
    - text (str): The input text to be augmented.
    - pos (str, optional): The part-of-speech tag. If provided, only synonyms of
      the specified part-of-speech will be considered (e.g., 'n' for nouns, 'v' for verbs).
    - custom_synonyms (dict, optional): A dictionary containing custom synonyms for words.
      Keys are words, and values are lists of custom synonyms.

    Returns:
    - str: The augmented text after lexical substitution.
    """
    words = word_tokenize(text)
    augmented_text = []
    stop_words = set(stopwords.words('english'))

    for word in words:
        if word.isalpha() and word not in stop_words:
            synonyms = get_synonyms(word, pos=pos, custom_synonyms=custom_synonyms)
            if synonyms:
                replacement = random.choice(synonyms)
                augmented_text.append(replacement)
            else:
                augmented_text.append(word)
        else:
            augmented_text.append(word)

    return ' '.join(augmented_text)

def augment_text_parallel(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in words if word.isalpha() and word not in stop_words]

    with ThreadPoolExecutor() as executor:
        augmented_tokens = list(executor.map(lexical_substitution, filtered_tokens))

    augmented_text = ' '.join(augmented_tokens)
    return augmented_text

def augment_chunk_parallel(chunk):
    """
    Perform parallel text augmentation on a given chunk of text.

    This function uses a ThreadPoolExecutor to perform the augmentation in parallel,
    which can significantly speed up the process when dealing with large amounts of text.

    Parameters:
    - chunk (list[str]): A list of strings, where each string is a piece of text to be augmented.

    Returns:
    - list[str]: A list of strings, where each string is the augmented version of the corresponding input text.
    """
    with ThreadPoolExecutor() as executor:
        augmented_chunk = list(executor.map(augment_text_parallel, chunk))
    return augmented_chunk
