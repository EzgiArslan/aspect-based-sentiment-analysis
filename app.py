from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel
from simpletransformers.ner import NERModel
from annotated_text import annotated_text
import streamlit as st


LABELS = ["servis", "ambians", "yemek", "restoran", "içecek", "lokasyon"]

DETAILED_LABELS = ['RESTAURANT#PRICES', 'AMBIENCE#GENERAL', 'RESTAURANT#GENERAL', 'FOOD#PRICES',
                   'RESTAURANT#MISCELLANEOUS', 'FOOD#QUALITY', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
                   'DRINKS#PRICES', 'LOCATION#GENERAL', 'FOOD#STYLE_OPTIONS', 'SERVICE#GENERAL']

DETAILED_LABELS_EN_TO_TR = {'RESTAURANT#PRICES': 'RESTORAN#FİYATLAR',
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

LABELS_EN_TO_TR = {"SERVICE": "SERVİS",
                   "AMBIENCE": "AMBİYANS",
                   "FOOD": "YEMEK",
                   "RESTAURANT": "RESTORAN",
                   "DRINKS": "İÇECEKLER",
                   "LOCATION": "LOKASYON"}

tags = ["O", "B-SERVICE", "I-SERVICE", "B-AMBIENCE", "I-AMBIENCE", "B-FOOD", "I-FOOD",
        "B-RESTAURANT", "I-RESTAURANT", "B-DRINKS", "I-DRINKS", "B-LOCATION", "I-LOCATION"]

tag_colors = {"SERVICE": "#8ef",
              "AMBIENCE": "#faa",
              "FOOD": "#afa",
              "RESTAURANT": "#fea",
              "DRINKS": "#eae",
              "LOCATION": "#eaf"}

POLARITY = {0: 'NEGATIVE',
            1: 'NEUTRAL',
            2: 'POSITIVE'}

EMOJIS_ASPECT = {'RESTAURANT#PRICES': '💸',
                 'AMBIENCE#GENERAL': '✨',
                 'RESTAURANT#GENERAL': '👨‍🍳',
                 'FOOD#PRICES': '💸',
                 'RESTAURANT#MISCELLANEOUS': '👨‍🍳',
                 'FOOD#QUALITY': '🍲',
                 'DRINKS#QUALITY': '🥤',
                 'DRINKS#STYLE_OPTIONS': '🥤',
                 'DRINKS#PRICES': '💸',
                 'LOCATION#GENERAL': '🗺️',
                 'FOOD#STYLE_OPTIONS': '🍲',
                 'SERVICE#GENERAL': '🍽️'}

EMOJIS_POLARITY = {'NEGATIVE': '😞',
                   'NEUTRAL': '😐',
                   'POSITIVE': '😀'}


@st.cache(allow_output_mutation=True)
def load_aspect_classification_model():
    model = MultiLabelClassificationModel(
        "bert", "aspect_classification_model", use_cuda=False
    )
    return model


@st.cache(allow_output_mutation=True)
def load_polarity_classification_model():
    model = ClassificationModel(
        "bert", "polarity_detection_model", use_cuda=False
    )
    return model


@st.cache(allow_output_mutation=True)
def load_aspect_extraction_model():
    model = NERModel("bert", "aspect_extraction_model", labels=tags, use_cuda=False)
    return model


def main():
    aspect_classification_model = load_aspect_classification_model()
    #polarity_model = load_polarity_classification_model()
    #aspect_extraction_model = load_aspect_extraction_model()

    st.title("Aspect Based Sentiment Analysis")
    _, col2, _ = st.beta_columns([1, 6, 1])
    with col2:
        st.image("images/word_cloud_plate.png")
    input_sentence = st.text_area('Input sentence for evaluation').strip()
    run = st.button('Evaluate')

    if run:
        polarities = []
        with st.spinner('Predicting...'):
            # Aspect extraction
            # extracted_aspects, raw_outputs = aspect_extraction_model.predict([input_sentence])
            # edited_sentence_test = []
            # text_with_tag = ""
            # tag_name = ""
            # for ext_aspect, word in zip(extracted_aspects[0], input_sentence.split(' ')):
            #     if ext_aspect[word] == "O":
            #         if text_with_tag != "":
            #             edited_sentence_test.append((f"{text_with_tag}",
            #                                          f"{LABELS_EN_TO_TR[tag_name]}",
            #                                          f"{tag_colors[tag_name]}"))
            #             text_with_tag = ""
            #         edited_sentence_test.append(f" {word} ")
            #     else:
            #         text_with_tag += f" {word} "
            #         tag_name = ext_aspect[word].split('-')[1]

            # Aspect classification
            aspect_predictions, aspect_model_outputs = aspect_classification_model.predict([input_sentence])
            aspects = [DETAILED_LABELS[i] for i in range(len(aspect_predictions[0])) if aspect_predictions[0][i]]

            # # Polarity detection
            # for pred in aspects:
            #     polarity_predictions, polarity_model_outputs = polarity_model.predict([input_sentence,
            #                                                                            DETAILED_LABELS_EN_TO_TR[pred]])
            #     polarities.append(POLARITY[polarity_predictions[0]])

        # st.markdown("### General Aspects")
        # if len(aspects) == 0:
        #     st.write("Out of Scope")
        # for aspect, polarity in zip(aspects, polarities):
        #     st.write("Aspect: ", EMOJIS_ASPECT[aspect], ' - '.join(DETAILED_LABELS_EN_TO_TR[aspect].split('#')))
        #     st.write("Polarity: ", EMOJIS_POLARITY[polarity], polarity)
        #
        # st.markdown("### Sentence")
        # annotated_text(*edited_sentence_test)
        st.write(aspects)

if __name__ == '__main__':
    main()
