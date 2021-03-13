from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer, Sentence
from MorphologicalDisambiguation.DisambiguationCorpus import DisambiguationCorpus
from MorphologicalDisambiguation.RootFirstDisambiguation import RootFirstDisambiguation
from trnlp import TrnlpToken
from turkish.deasciifier import Deasciifier


def tokenizer(reviews):
	for review in reviews:
		for sentence in review.sentences:
			tr_tokenizer = TrnlpToken()
			tr_tokenizer.settext(sentence.text.lower())
			tr_tokenizer = (tr_tokenizer.clean_stopwords(tr_tokenizer.clean_punch()))  # Hem durak kelimeler hem de noktalamaları temizlemiş olduk.
			sentence.text_tokens = [item[0] for item in tr_tokenizer]
	return reviews


def deascifiier(reviews):
	for review in reviews:
		for sentence in review.sentences:
			sentence.text_tokens = [Deasciifier(token).convert_to_turkish() for token in sentence.text_tokens]
	return reviews


def morphological_disambiguation(reviews):
	morphologicalDisambiguator = RootFirstDisambiguation()
	corpus = DisambiguationCorpus("penn_treebank.txt")
	morphologicalDisambiguator.train(corpus)
	morphologicalDisambiguator.saveModel()
	fsm = FsmMorphologicalAnalyzer()
	for review in reviews:
		for sentence in review.sentences:
			cur_sentence = Sentence(" ".join(sentence.text_tokens))
			fsmParseList = fsm.robustMorphologicalAnalysis(cur_sentence)
			candidateParses = morphologicalDisambiguator.disambiguate(fsmParseList)
			sentence.morphological_tokens = [str(token) for token in candidateParses]
			print(sentence.morphological_tokens)
	return reviews

# TODO: Dependency Parser eklenecek