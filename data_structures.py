# These classes helps to organize XML formatted SemEval data
class Review:
    def __init__(self, id, sentences):
        self.id = id
        self.sentences = sentences


class Sentence:
    def __init__(self, id, text, is_out_of_source, opinions):
        self.id = id
        self.text = text
        self.is_out_of_source = is_out_of_source
        self.opinions = opinions


class Opinion:
    def __init__(self, target, category, polarity, start, end):
        self.target = target
        self.category = category
        self.polarity = polarity
        self.start = start
        self.end = end
