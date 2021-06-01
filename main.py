# Custom classes and functions
from model_training import (aspect_extraction_with_cat, evaluate_cat, polarity_detection,
                            aspect_classification, aspect_extraction)
from data_preprocess import get_reviews, process_reviews

# TASKS
# ASPECT_CLASSIFICATION
# POLARITY_CLASSIFICATION
# CAT_ASPECT_EXTRACTION
# ASPECT_EXTRACTION
TASK = "ASPECT_EXTRACTION"


def main():
    # Reads Data
    train_reviews = get_reviews('input_files\\restaurant_train.xml')
    validation_reviews = get_reviews('input_files\\restaurant_validation.xml')
    train_reviews.extend(validation_reviews)
    test_reviews = get_reviews('input_files\\restaurant_test.xml')

    if TASK == "ASPECT_CLASSIFICATION":
        # Aspect Classification (Multilabel)
        target_labels = ['RESTAURANT#PRICES', 'AMBIENCE#GENERAL', 'RESTAURANT#GENERAL', 'FOOD#PRICES',
                         'RESTAURANT#MISCELLANEOUS', 'FOOD#QUALITY', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
                         'DRINKS#PRICES', 'LOCATION#GENERAL', 'FOOD#STYLE_OPTIONS', 'SERVICE#GENERAL']

        model = aspect_classification(train_reviews=train_reviews,
                                      test_reviews=test_reviews,
                                      target_labels=target_labels,
                                      output_path="aspect_classification_model")

    elif TASK == "POLARITY_CLASSIFICATION":
        # Polarity Classification
        target_labels_en_to_tr = {'RESTAURANT#PRICES': 'RESTORAN#FİYATLAR',
                                  'AMBIENCE#GENERAL': 'AMBİYANS#GENEL',
                                  'RESTAURANT#GENERAL': 'RESTORAN#GENEL',
                                  'FOOD#PRICES': 'YEMEK#FİYATLAR',
                                  'RESTAURANT#MISCELLANEOUS': 'RESTORAN#ÇEŞİTLİ',
                                  'FOOD#QUALITY': 'YEMEK#KALİTE',
                                  'DRINKS#QUALITY': 'İÇECEKLER#KALİTE',
                                  'DRINKS#STYLE_OPTIONS': 'İÇECEKLER#STİL_SEÇENEKLER',
                                  'DRINKS#PRICES': 'İÇECEKLER#FİYATLAR',
                                  'LOCATION#GENERAL': 'LOKASYON#GENEL',
                                  'FOOD#STYLE_OPTIONS': 'YEMEK#STİL_SEÇENEKLER',
                                  'SERVICE#GENERAL': 'SERVİS#GENEL'}

        model = polarity_detection(train_reviews=train_reviews,
                                   test_reviews=test_reviews,
                                   target_labels=target_labels_en_to_tr,
                                   output_path="polarity_detection_model")

    elif TASK == "ASPECT_EXTRACTION":
        # Aspect Extraction
        #Name entity recognition
        tags = ["O", "B-SERVICE", "I-SERVICE", "B-AMBIENCE", "I-AMBIENCE", "B-FOOD", "I-FOOD",
                "B-RESTAURANT", "I-RESTAURANT", "B-DRINKS", "I-DRINKS", "B-LOCATION", "I-LOCATION"]

        model = aspect_extraction(train_reviews=train_reviews,
                                  test_reviews=test_reviews,
                                  tags=tags,
                                  output_path="aspect_extraction_model")

    elif TASK == "CAT_ASPECT_EXTRACTION":
        # Data Preprocessing operations
        train_reviews, train_corpus, train_noun_counts = process_reviews(train_reviews,
                                                                         ignored_tags=["PUNCT", "SYM", "X", "AUX",
                                                                                       "CCONJ", "INTJ", "NUM", "PART"],
                                                                         include_corpus=True,
                                                                         include_noun_counts=True,
                                                                         remove_stopwords=True)

        sentences = [sentence.processed_text.split() for review in train_reviews for sentence in review.sentences]

        labels = {"servis", "ambians", "yemek", "restoran", "içecek", "lokasyon"}

        predicted_aspects = aspect_extraction_with_cat(sentences, labels, train_corpus, train_noun_counts, verbose=True)

        accuracy = evaluate_cat(reviews=train_reviews,
                                review_labels=["SERVICE", "AMBIENCE", "FOOD", "RESTAURANT", "DRINKS", "LOCATION"],
                                predicted_aspects=predicted_aspects)
        print(accuracy)


if __name__ == '__main__':
    main()
