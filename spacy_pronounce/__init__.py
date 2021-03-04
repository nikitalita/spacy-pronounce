# -*- coding: utf-8 -*-
from typing import Optional, List
from spacy.tokens import Token, Doc
from spacy.language import Language
import pyphen
import cmudict
from .g2p_en.expand import normalize_numbers
from .g2p_en.g2p import G2p
import unicodedata


@Language.factory("graphemes",
                  assigns=["token._.graphemes"],
                  default_config={"lang": None},
                  requires=["token.text", "token.pos"]
                  )
def make_spacy_pronounce(
        nlp: Language,
        name: str,
        lang: Optional[str]
):
    return SpacyPronounce(nlp, name, lang=lang)


class Graphemes:
    """
    whole_word
        text
        graphemes
    syllables:
        [syllable_text
         grapheme]
    """


class SpacyPronounce:
    def __init__(self, nlp: Language, name: str = "graphemes", lang: Optional[str] = None):

        """
        nlp: an instance of spacy
        name: defaults to "graphemes".
        lang: Optional, can be any format like : ["en", "en-us", "en_us", "en-US", ...]
              By default, it uses the language code of the model loaded.
        usage:
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("graphemes", after="tagger", config={"lang": "en_US"})
        """
        self.name = name
        self.g2p = G2p()
        self.cmu = cmudict.dict()

        self.vowel_phonemes = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                               'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                               'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                               'EY2',
                               'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'OW0', 'OW1',
                               'OW2', 'OY0', 'OY1', 'OY2',
                               'UH0', 'UH1', 'UH2', 'UW',
                               'UW0', 'UW1', 'UW2']
        lang = lang or nlp.lang
        lang, *country_code = lang.lower().replace("-", "_").split("_")
        if country_code:
            country_code = country_code[0].upper()
            lang = f"{lang}_{country_code}"
        elif lang == "en":
            lang = "en_US"
        elif lang == "de":
            lang = "de_DE"
        elif lang == "pt":
            lang = "pt_PT"

        try:
            self.syllable_dic = pyphen.Pyphen(lang=lang)
        except KeyError:
            # Don't do syllables
            pass

        Token.set_extension("graphemes", default=None, force=True)
        Token.set_extension("syllables", default=None, force=False)
        Token.set_extension("syllables_count", default=None, force=False)

    def syllables(self, word: str) -> Optional[list[str]]:
        if self.is_acronym(word):
            return [ch for ch in word]
        if word.isalpha():
            return self.syllable_dic.inserted(word.lower()).split("-")
        return None

    def is_acronym(self, word: str) -> bool:
        return word.strip('.').isupper()

    def normalize_word(self, word: str) -> str:
        word = normalize_numbers(word)
        word = ''.join(char for char in unicodedata.normalize('NFD', word)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        # handle acronyms
        word = word.replace(".", "")
        word = word.replace("-", "")
        word = word.replace("?", "")
        word = word.replace("!", "")
        word = word.replace("$", "")
        if word.isupper():
            word = '.'.join([ch for ch in word])
            word += '.'
        word = word.lower()
        return word

    def get_pronounciation(self, token: Token) -> object:
        graphemes = dict()
        graphemes["whole_word"] = None
        graphemes["syllables"] = None

        # normalize text for pronunciation checking
        word = self.normalize_word(token.text)
        print(word)

        pos = token.pos
        pron = None
        if word in self.g2p.homograph2features:  # Check homograph
            pron1, pron2, pos1 = self.g2p.homograph2features[word]
            if pos.startswith(pos1):
                pron = pron1
            else:
                pron = pron2
        elif word in self.cmu:  # lookup CMU dict
            pron = self.cmu[word][0]
        else:  # predict for oov
            pron = self.g2p.predict(word)
        graphemes["text"] = token.text
        graphemes["syllable_obj"] = token._.syllables
        graphemes["whole_word"] = pron
        if token._.syllables:
            graphemes["syllables"] = dict()
            if token._.syllables_count == 1:
                graphemes["syllables"][token.text] = pron
            else:
                sylprons = dict()
                pronidx = 0
                syllables: Optional[List[str]] = token._.syllables
                for syl in syllables:
                    vowels = 0
                    sylpron = []
                    while pronidx < len(pron):
                        phoneme = pron[pronidx]
                        if self.vowel_phonemes.__contains__(phoneme):
                            vowels += 1
                        if vowels > 1:
                            break
                        sylpron.append(phoneme)
                        pronidx += 1
                    if syllables.index(syl) == len(syllables) - 1 and pronidx < len(pron):
                        length = len(pron) - pronidx;
                        for i in range(pronidx, pronidx + length):
                            sylpron.append(pron[i])
                    sylprons[syl] = sylpron

                graphemes["syllables"] = sylprons

        return graphemes

    def __call__(self, doc: Doc):
        token: Token
        for token in doc:
            if hasattr(token._, 'syllables') and not token._.syllables:
                syllables = self.syllables(token.text)
                if syllables:
                    token._.set("syllables", syllables)
                    token._.set("syllables_count", len(syllables))
            graphemes = self.get_pronounciation(token)
            token._.set("graphemes", graphemes)

        return doc
