# Custom classes and functions
from data_preprocess import entity_aspect_to_sentence, iob_tagger
from cat import get_scores, rbf_attention
from utils import create_word_embeddings

# NLP tools
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel
from simpletransformers.ner import NERModel
from reach import Reach

# Default libraries
from collections import Counter

# Other libraries
import pandas as pd
import numpy as np
import torch

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score
from skmultilearn.utils import measure_per_label
from scipy.sparse import csr_matrix

CUDA = torch.cuda.is_available()
def aspect_extraction_with_cat(sentences, labels, corpus, nouns, vector_size=300, verbose=False):
    """
    Aspect extraction with cat (https://github.com/clips/cat)

    :param sentences: list of sentences
    :type sentences: list of string
    :param labels: set of labels
    :type labels: set of string
    :param corpus: list of lists of words
    :type corpus: list of list
    :param nouns: noun and their counts
    :type nouns: dictionary
    :param vector_size: word embeddings vector size, defaults to 300
    :type vector_size: integer
    :param verbose: verbose option, defaults to False
    :type verbose: boolean
    :return: aspects
    :rtype: list of list
    """
    create_word_embeddings(corpus, vector_size=vector_size, embeddings_path="embeddings/in_domain_vectors.vec")
    word_embeddings = Reach.load("embeddings/in_domain_vectors.vec",
                                 unk_word="<UNK>")

    new_nouns = Counter()
    for noun, count in nouns.items():
        if noun in word_embeddings.items:
            new_nouns[noun] += count

    new_nouns, _ = zip(*sorted(new_nouns.items(), key=lambda x: x[1], reverse=True))
    aspect_words = [[i] for i in new_nouns]
    aspect_words = aspect_words[:vector_size]

    scores = get_scores(sentences, aspect_words, word_embeddings, labels,
                        gamma=0.03, remove_oov=False, attention_func=rbf_attention)

    if verbose:
        labels_ = ["servis", "ambians", "yemek", "restoran", "içecek", "lokasyon"]
        for index, element in enumerate(scores):
            print(f"Sentence:\t{' '.join(sentences[index])}\n"
                  f"Aspects:\t{labels_[np.argmax(element)]}\n"
                  f"Prediction:\t{element}\n"
                  f"{'-' * 100}")

    aspects = []
    for score in scores:
        temp = [0] * 6
        temp[np.argmax(score)] = 1
        aspects.append(temp)
    return aspects


def evaluate_cat(reviews, review_labels, predicted_aspects):
    """
    Calculates accuracy for predicted aspects from cat algorithm

    :param reviews: list of reviews
    :type reviews: list of Review
    :param review_labels: review labels list
    :type review_labels: list of string
    :param predicted_aspects: predicted aspects
    :type predicted_aspects: list of list
    :return: accuracies for all labels
    :rtype: list of float
    """
    categories = [sentence.categories for review in reviews for sentence in review.sentences]
    actual = []
    for category in categories:
        temp = [0] * 6
        for i in category:
            temp[review_labels.index(i)] = 1
        actual.append(temp)

    return measure_per_label(accuracy_score, csr_matrix(actual), csr_matrix(predicted_aspects))


def bert_multiclass_classification(train_df, model_args=None, output_path="multiclass_classification_model"):
    """
    BERT multiclass classification model training function
    For more information: https://simpletransformers.ai/docs/multi-class-classification/
    dbmdz/bert-base-turkish-cased used for Turkish BERT Model
    https://huggingface.co/dbmdz/bert-base-turkish-cased

    :param train_df: training data
    :type train_df: pd.DataFrame
    :param model_args: model arguments
    :type model_args: dictionary
    :output path: output path to save model
    :type output_path: string
    :return: trained BERT model
    :rtype: ClassificationModel
    """
    if model_args is None:
        model_args = {
            "use_early_stopping": True,
            "early_stopping_delta": 0.01,
            "early_stopping_metric": "mcc",
            "early_stopping_metric_minimize": False,
            "early_stopping_patience": 5,
            "evaluate_during_training_steps": 1000,
            "fp16": False,
            "num_train_epochs": 5,
            "output_dir": output_path
        }

    model = ClassificationModel("bert",
                                "dbmdz/bert-base-turkish-cased",
                                use_cuda=CUDA,
                                args=model_args,
                                num_labels=train_df.labels.nunique())

    model.train_model(train_df, acc=accuracy_score)

    return model


def bert_multilabel_classification(train_df, model_args=None, output_path="multilabel_classification_model"):
    """
    BERT multilabel classification model training function
    For more information: https://simpletransformers.ai/docs/multi-label-classification/
    dbmdz/bert-base-turkish-cased used for Turkish BERT Model
    https://huggingface.co/dbmdz/bert-base-turkish-cased

    :param train_df: training data
    :type train_df: pd.DataFrame
    :param model_args: model arguments
    :type model_args: dictionary
    :output path: output path to save model
    :type output_path: string
    :return: trained BERT model
    :rtype: MultiLabelClassificationModel
    """
    if model_args is None:
        model_args = {
            "use_early_stopping": True,
            "early_stopping_delta": 0.01,
            "early_stopping_patience": 5,
            "evaluate_during_training_steps": 1000,
            "evaluate_during_training_verbose": True,
            "fp16": False,
            "num_train_epochs": 5,
            "output_dir": output_path
        }

    model = MultiLabelClassificationModel("bert",
                                          "dbmdz/bert-base-turkish-cased",
                                          use_cuda=CUDA,
                                          args=model_args,
                                          num_labels=len(train_df.labels[0]))

    model.train_model(train_df, acc=accuracy_score)

    return model


def bert_ner_model(train_df, tags, model_args=None, output_path="ner_model"):
    """
    BERT NER model training function
    For more information: https://simpletransformers.ai/docs/ner-model/
    dbmdz/bert-base-turkish-cased used for Turkish BERT Model
    https://huggingface.co/dbmdz/bert-base-turkish-cased

    :param train_df: training data
    :type train_df: pd.DataFrame
    :param tags: Custom IOB tags
    :type tags: list of string
    :param model_args: model arguments
    :type model_args: dictionary
    :output path: output path to save model
    :type output_path: string
    :return: trained BERT NER model
    :rtype: NERModel
    """
    if model_args is None:
        model_args = {
            "use_early_stopping": True,
            "early_stopping_delta": 0.01,
            "early_stopping_patience": 5,
            "evaluate_during_training_steps": 1000,
            "evaluate_during_training_verbose": True,
            "fp16": False,
            "num_train_epochs": 5,
            "output_dir": output_path
        }

    model = NERModel("bert",
                     "dbmdz/bert-base-turkish-cased",
                     labels=tags,
                     use_cuda=CUDA,
                     args=model_args)

    model.train_model(train_df)

    return model


def polarity_detection(train_reviews, test_reviews, target_labels, output_path):
    """
    Performs polarity classification with BERT (Sentence-pair classification)
    For more information: https://simpletransformers.ai/docs/sentence-pair-classification/

    :param train_reviews: list of customer reviews
    :type train_reviews: list of Review
    :param test_reviews: list of customer reviews
    :type test_reviews: list of Review
    :param target_labels: target labels for classification
    :type target_labels: dictionary
    :param output_path: output path to save model
    :type output_path: string
    :return: trained BERT model
    :rtype: ClassificationModel
    """
    # Data Preprocess for model training
    train = entity_aspect_to_sentence(train_reviews, target_labels)
    train.labels = pd.factorize(train.labels, sort=True)[0]

    test = entity_aspect_to_sentence(test_reviews, target_labels)
    test.labels = pd.factorize(test.labels, sort=True)[0]

    # Model Training
    model = bert_multiclass_classification(train, output_path=output_path)

    # Evaluation
    result, model_outputs, wrong_predictions = model.eval_model(test)

    overall_accuracy = accuracy_score(test.labels.values, model_outputs.argmax(axis=1))

    print(f"\nAccuracy:\n{overall_accuracy}")

    cls_report = classification_report(test.labels.values, model_outputs.argmax(axis=1))

    print(f"\nClassification Report:\n{cls_report}")

    return model


def aspect_classification(train_reviews, test_reviews, target_labels, output_path):
    """
    Performs aspect classification with BERT

    :param train_reviews: list of customer reviews
    :type train_reviews: list of Review
    :param test_reviews: list of customer reviews
    :type test_reviews: list of Review
    :param target_labels: target labels for classification
    :type target_labels: list of string
    :param output_path: output path to save model
    :type output_path: string
    :return: trained BERT model
    :rtype: MultiLabelClassificationModel
    """
    train_sentences = [sentence.text for review in train_reviews for sentence in review.sentences if
                       not sentence.is_out_of_source]

    train_categories = [sentence.categories for review in train_reviews for sentence in review.sentences if
                        not sentence.is_out_of_source]

    test_sentences = [sentence.text for review in test_reviews for sentence in review.sentences if
                      not sentence.is_out_of_source]

    test_categories = [sentence.categories for review in test_reviews for sentence in review.sentences if
                       not sentence.is_out_of_source]

    train_labels = []
    for category in train_categories:
        temp = [0] * len(target_labels)
        for i in category:
            temp[target_labels.index(i)] = 1
        train_labels.append(temp)

    test_labels = []
    for category in test_categories:
        temp = [0] * len(target_labels)
        for i in category:
            temp[target_labels.index(i)] = 1
        test_labels.append(temp)

    train = pd.DataFrame(zip(train_sentences, train_labels), columns=['text', 'labels'])

    test = pd.DataFrame(zip(test_sentences, test_labels), columns=['text', 'labels'])

    model = bert_multilabel_classification(train, output_path=output_path)

    result, model_outputs, wrong_predictions = model.eval_model(test)

    overall_accuracy = accuracy_score(np.array([i for i in test.labels.values]), model_outputs.round())

    print(f"\nAccuracy:\n{overall_accuracy}")

    label_based_accuracy = pd.DataFrame(measure_per_label(accuracy_score,
                                                          csr_matrix(np.array([i for i in test.labels.values])),
                                                          csr_matrix(model_outputs.round())), index=target_labels)

    print(f"\nLabel Based Accuracy:\n{label_based_accuracy}")

    cls_report = classification_report(np.array([i for i in test.labels.values]),
                                       model_outputs.round())

    print(f"\nClassification Report:\n{cls_report}")

    return model


def aspect_extraction(train_reviews, test_reviews, tags, output_path):
    """
    Perform aspect extraction using BERT NER Model
    For more information: https://simpletransformers.ai/docs/ner-specifics/

    :param train_reviews: list of customer reviews
    :type train_reviews: list of Review
    :param test_reviews: list of customer reviews
    :type test_reviews: list of Review
    :param tags: Custom IOB Tags
    :type tags: list of string
    :param output_path: output path to save model
    :type output_path: string
    :return: trained BERT NER model
    :rtype: NERModel
    """
    train_df = iob_tagger(train_reviews, tags)

    test_df = iob_tagger(test_reviews, tags)

    model = bert_ner_model(train_df, tags, output_path=output_path)

    result, model_outputs, preds_list = model.eval_model(test_df)

    print(result)

    return model
