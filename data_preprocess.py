# Custom classes and functions
from data_structures import Review, Sentence, Opinion

# NLP tools
from turkish.deasciifier import Deasciifier
from nltk.corpus import stopwords
import nltk as nltk
import stanza

# Other Libraries
import pandas as pd

# Default libraries
from xml.etree import cElementTree as ET
from collections import defaultdict
import os

# Initialize configs and download required files
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
nltk.download('stopwords')
stanza.download("tr")


def deascifiier(reviews):
    """
    This function helps to correct bad or wrong type of sentences
    Example:
        Nasilsiniz? -> Nasılsınız?

    :param reviews: list of reviews for given dataset
    :type reviews: list of Review
    :return: corrected reviews
    :rtype: list of Review
    """
    for review in reviews:
        for sentence in review.sentences:
            sentence.text = Deasciifier(sentence.text).convert_to_turkish()
            sentence.text = sentence.text.lower()
            for op in sentence.opinions:
                op.target = Deasciifier(op.target).convert_to_turkish()
                op.target = op.target.lower()
    return reviews


def get_reviews(file_name):
    """
    Reads XML formatted input files and convert them into our Data Structures (Review, Sentence, Opinion)

    :param file_name: input file name
    :type file_name: string
    :return: reviews
    :rtype: list of Review
    """
    tree = ET.parse(file_name)
    root = tree.getroot()
    reviews = []
    for rid in root:
        current_review = Review(rid.attrib['rid'], [])
        for sentence in rid.iter('sentences'):
            for sentences in sentence.iter('sentence'):
                if 'OutOfScope' in sentences.attrib:  # Eliminates OutOfScope samples
                    current_sentence = Sentence(sentences.attrib['id'], list(sentences.iter('text'))[0].text, True, [])
                else:
                    current_sentence = Sentence(sentences.attrib['id'], list(sentences.iter('text'))[0].text, False, [])
                    categories = []
                    polarities = []
                    for opinion in sentences.iter('Opinion'):
                        op = Opinion(opinion.attrib['target'], opinion.attrib['category'], opinion.attrib['polarity'],
                                     int(opinion.attrib['from']), int(opinion.attrib['to']))
                        categories.append(op.category)
                        polarities.append(op.polarity)
                        current_sentence.opinions.append(op)
                    current_sentence.categories = categories
                    current_sentence.polarities = polarities
                current_review.sentences.append(current_sentence)
        reviews.append(current_review)
    return reviews


def process_reviews(reviews,
                    processors='tokenize,lemma,mwt,pos',
                    ignored_tags=None,
                    include_noun_counts=True,
                    include_corpus=True,
                    remove_stopwords=True):
    """
    Processes reviews with Stanza and returns processed version of reviews
    and if requested returns corpus or noun counts

    :param reviews: list of reviews
    :type reviews: list of Review
    :param processors: selected processors for Stanza pipeline (must be in string format and comma-separated),
    defaults to 'tokenize,lemma,mwt,pos'
    :type processors: string
    :param ignored_tags: ignored pos tags (https://universaldependencies.org/u/pos/), defaults to None
    :type ignored_tags: list of string
    :param include_noun_counts: selection of whether return parameters include noun counts or not, defaults to True
    :type include_noun_counts: boolean
    :param include_corpus: selection of whether return parameters include corpus or not, defaults to True
    :type include_corpus: boolean
    :param remove_stopwords: selection of whether remove stopwords or not, defaults to True
    :type remove_stopwords: boolean
    :returns: (processed reviews, corpus, noun counts)
    :rtype: (list of Review, list of list, defaultdict)
    """
    # Initializes variables for selections
    if ignored_tags is None:
        ignored_tags = []
    noun_counts = defaultdict(int) if include_noun_counts else None
    corpus = [] if include_corpus else None

    reviews = deascifiier(reviews)
    # Initializes stanza for preprocessing
    nlp = stanza.Pipeline(lang='tr', processors=processors)
    # Gets stop words
    stop_words = set(stopwords.words('turkish')) if remove_stopwords else set()

    for review in reviews:
        for sentence in review.sentences:
            # Each sentence is sent to pipeline for preprocessing
            processed_sentence = nlp(sentence.text)
            # Traverses each word with pos-tagging
            processed_sentence_words = []
            temp_corpus = []
            for word in processed_sentence.sentences[0].words:
                w = word.lemma.lower()
                if include_noun_counts and word.upos in ["NOUN"]:
                    noun_counts[w] += 1
                if word.upos not in ignored_tags and w not in stop_words:
                    processed_sentence_words.append(w)
                    if include_corpus and w not in temp_corpus:
                        temp_corpus.append(w)
            if include_corpus:
                corpus.append(temp_corpus)
            # Adds processed text to sentence object
            sentence.processed_text = ' '.join(processed_sentence_words)
    return reviews, corpus, noun_counts


def entity_aspect_to_sentence(reviews, target_labels):
    """
    Transforms ENTITY#ASPECT's to sentences

    :param reviews: list of reviews
    :type reviews: list of Review
    :param target_labels: target label translator dict
    :type target_labels: dictionary
    :return: preprocessed reviews for sentence-pair classification
    :rtype: pd.DataFrame
    """
    # Creates sentences from categories
    # We have categories like ENTITY#ASPECT in our data
    # We can express this categories entity, aspect
    # In example: FOOD#STYLE_OPTIONS -> food, style options
    # In our case we translate this sentence to TR -> yemek, stil seçenekler
    sentences_part1 = []
    sentences_part2 = []
    categories_all = []
    for review in reviews:
        for sentence in review.sentences:
            if not sentence.is_out_of_source:
                sentence_cat = []
                for category, polarity in zip(sentence.categories, sentence.polarities):
                    cur_category = target_labels[category]
                    if cur_category not in sentence_cat:
                        temp = cur_category.lower().split('#')
                        sentences_part1.append(sentence.text)
                        sentences_part2.append(f"{temp[0]}, {' '.join(temp[1].split('_'))}")
                        categories_all.append(polarity)
                        sentence_cat.append(cur_category)

    return pd.DataFrame(zip(sentences_part1, sentences_part2, categories_all), columns=["text_a", "text_b", "labels"])


def iob_tagger(reviews, tags):
    """
    Performs IOB Tagging

    :param reviews: list of reviews
    :type reviews: list of Review
    :param tags: IOB Custom Labels
    :type tags: list
    :return: preprocessed reviews for NER model
    :rtype: pd.DataFrame
    """
    sentence_ids, words, labels = [], [], []
    sentence_id = 0
    for review in reviews:
        for sentence in review.sentences:
            if not sentence.is_out_of_source:
                temp_text = sentence.text
                for opinion in sentence.opinions:
                    if opinion.target == "NULL":
                        continue
                    text_parts = temp_text[opinion.start:opinion.end].split(' ')
                    entity = opinion.category.split('#')[0]
                    new_parts = [f"B-{entity}"]
                    for part in range(1, len(text_parts)):
                        new_parts.append(f"I-{entity}")
                    temp_text = temp_text[:opinion.start] + ' '.join(new_parts) + temp_text[opinion.end:]
                temp_text_parts = temp_text.split(' ')
                original_text_parts = sentence.text.split(' ')
                for temp_text_part, original_text_part in zip(temp_text_parts, original_text_parts):
                    sentence_ids.append(sentence_id)
                    words.append(original_text_part)
                    if temp_text_part not in tags:
                        labels.append("O")
                    else:
                        labels.append(temp_text_part)
            sentence_id += 1
    return pd.DataFrame(zip(sentence_ids, words, labels), columns=["sentence_id", "words", "labels"])
