# -*- coding: utf-8 -*-
from typing import Optional, Any
from spacy.tokens import Token, Doc
from spacy.language import Language
import pyphen
import cmudict
from .g2p_en.expand import normalize_numbers
from .g2p_en.g2p import G2p
from .g2p_en.data.contractions import ENGLISH_CONTRACTIONS

import unicodedata


@Language.factory("phonemes",
                  assigns=["token._.phonemes", "token._.syllables", "token._.syllables_count"],
                  default_config={
                      "lang": None,
                      "syllable_grouping": True,
                      "consonant_dupe": True,
                      "sound_out_acronyms": False,
                      "pronounce_punctuation": False,
                      "fix_split_contractions": True
                  },
                  requires=["token.text", "token.pos"],

                  )
def make_spacy_pronounce(
        nlp: Language,
        name: str,
        lang: Optional[str],
        syllable_grouping: bool,
        consonant_dupe: bool,
        sound_out_acronyms: bool,
        pronounce_punctuation: bool,
        fix_split_contractions: bool
):
    return SpacyPronounce(nlp, name,
                          lang=lang,
                          syllable_grouping=syllable_grouping,
                          consonant_dupe=consonant_dupe,
                          sound_out_acronyms=sound_out_acronyms,
                          pronounce_punctuation=pronounce_punctuation,
                          fix_split_contractions= fix_split_contractions
                          )


class Graphemes:
    """
    whole_word
        text
        graphemes
    syllables:
        [syllable_text
         grapheme]
    """


class FakeToken(object):
    def __init__(self):
        pass

class SpacyPronounce:
    def __init__(self, nlp: Language, name: str = "phonemes",
                 lang: Optional[str] = None,
                 syllable_grouping: bool = True,
                 consonant_dupe: bool = True,
                 sound_out_acronyms: bool = False,
                 pronounce_punctuation: bool = False,
                 fix_split_contractions: bool = True):

        """
        nlp: an instance of spacy
        name: defaults to "phonemes".
        lang: Optional, can be any format like : ["en", "en-us", "en_us", "en-US", ...]
              By default, it uses the language code of the model loaded.
        usage:
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("phonemes", after="tagger", config={"lang": "en_US"})
        """
        self.name = name
        self.g2p = G2p()
        self.cmu = cmudict.dict()
        self.syllable_grouping = syllable_grouping
        self.consonant_dupe = consonant_dupe
        self.sound_out_acronyms = sound_out_acronyms
        self.pronounce_punctuation = pronounce_punctuation
        self.fix_split_contractions = fix_split_contractions

        self.vowel_phonemes = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                               'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                               'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                               'EY2',
                               'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'OW0', 'OW1',
                               'OW2', 'OY0', 'OY1', 'OY2',
                               'UH0', 'UH1', 'UH2', 'UW',
                               'UW0', 'UW1', 'UW2']
        self.consonants = "bcdfghjklmnpqrstvwxyz"
        lang = lang or nlp.lang
        lang, *country_code = lang.lower().replace("-", "_").split("_")
        if lang != "en":
            raise Exception("spacy_pronounce is only currently implemented for english")

        Token.set_extension("phonemes", default=None, force=True)
        if self.syllable_grouping:
            self.syllable_dic = pyphen.Pyphen(lang=lang)
            if not Token.has_extension("syllables"):
                Token.set_extension("syllables", default=None, force=False)
                Token.set_extension("syllables_count", default=None, force=False)

    def get_syllables(self, word: str) -> Optional[list[str]]:
        if word.replace("'", "").isalpha():
            return self.syllable_dic.inserted(word.lower()).split("-")
        return None

    @staticmethod
    def is_acronym(word: str) -> bool:
        return len(word) != 1 and word.strip('.').isupper()

    @staticmethod
    def normalize_word(word: str) -> str:
        word = normalize_numbers(word)
        word = ''.join(char for char in unicodedata.normalize('NFD', word)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        word = word.replace(".", "")
        word = word.replace(",", "")
        word = word.replace("-", "")
        word = word.replace("?", "")
        word = word.replace("!", "")
        word = word.replace("$", "")
        # handle acronyms
        if word.isupper():
            word = '.'.join([ch for ch in word])
            word += '.'
        word = word.lower()
        return word

    def group_syllables(self, token: Token, pron):
        sylprons = dict()
        if not token._.syllables:
            sylprons[token.text] = pron
            return sylprons
        if token._.syllables_count == 1:
            sylprons[token._.syllables[0]] = pron
            return sylprons
        pronidx = 0
        syllables: Optional[list[str]] = token._.syllables

        for sylidx in range(len(syllables)):
            syl = syllables[sylidx]
            vowels = 0
            sylpron = []
            while pronidx < len(pron):
                phoneme = pron[pronidx]
                syl_first_letter = syl[0].lower()
                if syl_first_letter == "'" and len(syl) > 1:
                    syl_first_letter = syl[1].lower()
                phoneme_first_letter = phoneme[0].lower()

                if self.vowel_phonemes.__contains__(phoneme):
                    vowels += 1

                # If a consonant phoneme and the syllable does not contain it, skip
                else:
                    consonant = phoneme[0].lower()
                    if vowels == 1 or syl.__contains__("'"):
                        if consonant == "k":
                            if not (syl.lower().__contains__("k") or syl.lower().__contains__("c")):
                                break
                        elif consonant == "z":
                            if not(syl.lower().__contains__("s") or syl.lower().__contains__("z")):
                                break
                        elif not syl.lower().__contains__(consonant):
                            break

                # get the consonant phoneme from the last syllable if necessary
                if self.consonant_dupe and len(sylpron) == 0 and sylidx > 0 \
                        and syl_first_letter != phoneme_first_letter \
                        and self.consonants.__contains__(syl_first_letter):
                    last_pron = sylprons[syllables[sylidx - 1]]
                    last_phoneme = last_pron[len(last_pron) - 1]
                    if last_phoneme[0].lower() == syl_first_letter:
                        sylpron.append(last_phoneme)

                if vowels > 1:
                    break
                sylpron.append(phoneme)
                pronidx += 1
            # append the rest to the last syllable if we missed some
            if syllables.index(syl) == len(syllables) - 1 and pronidx < len(pron):
                length = len(pron) - pronidx
                for i in range(pronidx, pronidx + length):
                    sylpron.append(pron[i])
            sylprons[syl] = sylpron
        return sylprons

    def get_pronunciation(self, token: Token or FakeToken) -> object:

        phonemes = dict()
        phonemes["whole_word"] = None
        phonemes["syllables"] = None

        # pronunciation checking
        if token.pos_ == "PUNCT":
            if self.pronounce_punctuation:
                word = token.text
            else:
                return phonemes
        else:
            # normalize text for pronunciation checking
            word = self.normalize_word(token.text)

        if word == "":
            return phonemes

        # if pronounce acronyms
        if self.sound_out_acronyms and self.is_acronym(token.text):
            word = '.'.join([ch for ch in word])
            word += '.'
            if self.syllable_grouping:
                acr_syls = [ch for ch in token.text.strip('.')]
                token._.set("syllables", acr_syls)
                token._.set("syllables_count", len(acr_syls))

        tag = token.tag_

        if word in self.g2p.homograph2features:  # Check homograph
            pron1, pron2, tag1 = self.g2p.homograph2features[word]
            if tag.startswith(tag1):
                pron = pron1
            else:
                pron = pron2
        elif word in self.cmu:  # lookup CMU dict
            pron = self.cmu[word][0]
        else:  # predict for oov
            pron = self.g2p.predict(word)
        phonemes["whole_word"] = pron
        if self.syllable_grouping or token.pos_ == "CONTRACTION":
            phonemes["syllables"] = self.group_syllables(token, pron)
        return phonemes

    def set_syllables(self, token: Token):
        if self.syllable_grouping and hasattr(token._, 'syllables') and not token._.syllables:
            syllables = self.get_syllables(token.text)
            if syllables:
                token._.set("syllables", syllables)
                token._.set("syllables_count", len(syllables))

    def __call__(self, doc: Doc):
        tokenIdx = 0
        while tokenIdx < len(doc):
            token: Token = doc[tokenIdx]
            # If we're not using a tokenizer that keeps the contractions together
            if self.fix_split_contractions and tokenIdx < len(doc) - 1 \
                    and not ENGLISH_CONTRACTIONS.__contains__(token.text.lower()) \
                    and len(doc[tokenIdx+1].text) > 1 \
                    and doc[tokenIdx+1].text.__contains__("'"):
                skip = self.handle_contraction(tokenIdx, doc)
                if skip > 0:
                    tokenIdx += skip
                    continue
                # If not a valid contraction, keep going
            self.set_syllables(token)
            phonemes = self.get_pronunciation(token)
            token._.set("phonemes", phonemes)
            tokenIdx += 1
        return doc

    def get_syllables_contraction(self, word: str) -> Optional[list[str]]:
        syllables = self.get_syllables(word)
        if syllables:
            return syllables
        if word.replace("'", "").isalpha():
            words = word.split("'")
            syllables = []
            for i in range(len(words)):
                syls = self.syllable_dic.inserted(words[i].lower()).split("-")
                if len(syls) > 0 and syls[0] != '':
                    syls.pop(0)
                if i > 0 and len(syls) > 0:
                    syls[0] = "'" + syls[0]
                syllables.extend(syls)
            return syllables
        return None

    def handle_contraction(self, tokenIdx: int, doc: Doc) -> int:
        tokens = [doc[tokenIdx]]
        idx = tokenIdx
        while idx < len(doc) - 1:
            idx += 1
            next_token = doc[idx]
            if next_token.text.__contains__("'"):
                tokens.append(next_token)
            else:
                break
        word = "".join([t.text for t in tokens])
        print(word)
        if ENGLISH_CONTRACTIONS.__contains__(word.lower()):
            new_token = FakeToken()
            new_token.text = word
            new_token.tag_ = ""
            new_token.pos_ = "CONTRACTION"
            new_token._ = FakeToken()
            new_token._.syllables = [t.text for t in tokens]
            new_token._.syllables_count = len(tokens)
            cont_phones = self.get_pronunciation(new_token)
            t: Token
            for t in tokens:
                t_phone = dict()
                t_phone["whole_word"] = cont_phones["syllables"][t.text]
                if self.syllable_grouping:
                    t_phone["syllables"] = dict()
                    t_phone["syllables"][t.text] = cont_phones["syllables"][t.text]
                    if t._.syllables == None:
                        t._.syllables = [t.text]
                        t._.syllables_count = 1
                t._.set("phonemes", t_phone)
            return len(tokens)
        return 0
