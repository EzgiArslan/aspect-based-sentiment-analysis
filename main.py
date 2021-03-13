from data_structures import Review, Sentence
from xml.etree import cElementTree as ET
from data_preprocess import tokenizer, deascifiier, morphological_disambiguation

if __name__ == '__main__':
    tree = ET.parse(r'input_files\restaurant_test_1.xml')
    root = tree.getroot()
    reviews=[]
    for rid in root:
        current_review = Review(rid.attrib['rid'],[])
        for sentence in rid.iter('sentences'):
            for sentences in sentence.iter('sentence'):
                if 'OutOfScope' in sentences.attrib:  # elimine ediyor
                    current_sentence = Sentence(sentences.attrib['id'],list(sentences.iter('text'))[0].text,True,None)
                else:
                    current_sentence = Sentence(sentences.attrib['id'],list(sentences.iter('text'))[0].text,False,[])
                    for opinion in sentences.iter('Opinion'):
                        current_sentence.opinions.append(opinion)
                current_review.sentences.append(current_sentence)
        reviews.append(current_review)

    reviews = morphological_disambiguation(deascifiier(tokenizer(reviews)))
    print(1)
