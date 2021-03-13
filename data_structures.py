class Review:
    def __init__(self,id,sentences):
        self.id = id
        self.sentences = sentences
class Sentence:
    def __init__(self,id,text,is_out_of_source,opinions):
        self.id = id
        self.text = text
        self.text_tokens=None
        self.morphological_tokens=None
        self.is_out_of_source = is_out_of_source
        self.opinions = opinions
