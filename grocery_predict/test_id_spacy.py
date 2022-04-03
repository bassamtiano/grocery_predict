import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.id import Indonesian
from spacy.lang.id.stop_words import STOP_WORDS

from spacy.lookups import Lookups

from spacy.pipeline import Tok2Vec
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from spacy.pipeline import Tagger

config = {"model": DEFAULT_TAGGER_MODEL}


nlp = Indonesian()



texts = ["Umumkan Awal Puasa 3 April, Menag: Kita Harap Umat Islam RI Puasa Bersama-sama", "Kali Bekasi yang berada di pintu air Bendung Bekasi, Jalan M. Hasibuan, Kelurahan Margajaya, Kecamatan Bekasi Selatan, Kota Bekasi, kembali memperlihatkan busa tebal berwarna putih"]

# for doc in tokenizer.pipe(texts, batch_size=50):
#     print(type(doc))

doc = nlp(texts[0])

config = {"model": DEFAULT_TAGGER_MODEL}
tagger = nlp.add_pipe("tagger", config = config)


tagger = Tagger(nlp.vocab, nlp)

lemma_list = list(LOOKUP.items())

print(lemma_list)