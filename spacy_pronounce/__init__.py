# -*- coding: utf-8 -*-
from typing import Optional
from spacy.tokens import Token, Doc
from spacy.language import Language
import pyphen
import cmudict

from eng_to_ipa import cmu_syllable_count
from .g2p_en.expand import normalize_numbers
from .g2p_en.g2p import G2p
from .g2p_en.data.contractions import ENGLISH_CONTRACTIONS
from .g2p_en.data.multiples import CMU_AMBIGUOUS_STRESS_WORDS
from contractions import contractions_dict

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
                          fix_split_contractions=fix_split_contractions
                          )


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
        self.hiatus = [["er", "iy"], ["iy", "ow"], ["uw", "ow"], ["iy", "ah"], ["iy", "ey"], ["uw", "eh"], ["er", "eh"]]
        self.vowel_phonemes = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                               'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                               'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                               'EY2',
                               'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'OW0', 'OW1',
                               'OW2', 'OY0', 'OY1', 'OY2',
                               'UH0', 'UH1', 'UH2', 'UW',
                               'UW0', 'UW1', 'UW2']
        self.affricate_phonemes = ['CH', 'JH']
        self.consonants = "bcdfghjklmnpqrstvwxyz"
        self.contractions = contractions_dict.keys()
        lang = lang or nlp.lang
        lang, *country_code = lang.lower().replace("-", "_").split("_")
        if lang != "en":
            raise Exception("spacy_pronounce is only currently implemented for english")

        Token.set_extension("phonemes", default=None, force=True)
        if self.syllable_grouping:
            if len(country_code) == 0:
                country_code.append("US")
            self.syllable_dic = pyphen.Pyphen(lang=lang +'-' + country_code[0])
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

    @staticmethod
    def find_first_vowel(word: str) -> int:
        for i in range(len(word)):
            if word[i] in "aeiou":
                return i
        return -1

    @staticmethod
    def should_break_to_next_syllable(self, curr_phone: str, next_phone: Optional[str], syl_has_vowel: bool) -> bool:
        if not syl_has_vowel:
            return False
        elif curr_phone in self.vowel_phonemes:
            return True
        elif next_phone:
            if next_phone in self.vowel_phonemes:
                return True
        return False

    @staticmethod
    def group_phone_syllables(self, phones: list[str]) -> list[list[str]]:
        groups = list[list[str]]()
        group = list[str]()
        has_vowel = False
        for i, phone in enumerate(phones):
            next_phone = None
            if i + 1 < len(phones):
                next_phone = phones[i + 1]
            if self.should_break_to_next_syllable(phone, next_phone, has_vowel):
                groups.append(group)
                group = list[str]().append(phone)
                has_vowel = False
            else:
                group.append(phone)

            if phone in self.vowel_phonemes:
                has_vowel = True
        return groups

    @staticmethod
    def strip_stresses_from_phones(phones: list[str]) -> list[str]:
        return [''.join(i.lower() for i in p if not i.isdigit()) for p in phones]


    def group_syllables(self, word: str, pron: list[str], syllables: Optional[list[str]]):

        sylprons = dict[str]()
        stripped_phones = self.strip_stresses_from_phones(pron)
        phone_syl_count = cmu_syllable_count(' '.join(stripped_phones))

        if syllables and len(syllables) == 1 and len(syllables) == phone_syl_count:
            sylprons[syllables[0]] = pron
            return sylprons
        pronidx = 0

        # phone_syl_groups = self.group_phone_syllables(pron)
        # Pyphen failed to syllablize properly, we need to redo it
        # if not syllables or len(syllables) != phone_syl_count:
        #    pass

        for sylidx in range(len(syllables)):
            syl = syllables[sylidx]
            vowels = 0
            sylpron = []
            next_syl = None
            if sylidx < len(syllables) - 1:
                next_syl = syllables[sylidx+1]
            while pronidx < len(pron):
                phoneme = pron[pronidx]
                next_phoneme = None
                prev_phoneme = None
                if (pronidx < len(pron) - 1):
                    next_phoneme = pron[pronidx+1]
                if (pronidx > 0):
                    prev_phoneme = pron[pronidx-1]
                syl_first_letter = syl[0].lower()
                if syl_first_letter == "'" and len(syl) > 1:
                    syl_first_letter = syl[1].lower()
                phoneme_first_letter = phoneme[0].lower()

                if phoneme in self.vowel_phonemes:
                    vowels += 1
                    if vowels > 1:
                        break
                # If a consonant phoneme and the syllable does not contain it, skip
                elif vowels == 1:

                    if next_phoneme and next_phoneme in self.vowel_phonemes:
                        # If we're not duplicating consonants, don't bother
                        if not self.consonant_dupe:
                            break
                        # If the previous phoneme was also a consonant, the two consonants have different sounds,
                        # don't dupe
                        if prev_phoneme not in self.vowel_phonemes:
                            break
                        if next_syl:
                            if syl[-1] != next_syl[0] or phoneme in self.affricate_phonemes:
                                break

                        # If consonant dupe...
                        # vowel_idx = self.find_first_vowel(syl)
                        # if "'" in syl and self.find_first_vowel(syl) < len(syl) - 2:
                        #     suffix = syl[vowel_idx + 1: len(syl)].lower()
                        #     if phoneme_first_letter == "k":
                        #         if not ("k" or "c" in suffix):
                        #             break
                        #     elif phoneme_first_letter == "z":
                        #         if not ("s" or "z" in suffix):
                        #             break
                        #     elif phoneme_first_letter not in suffix:
                        #         break
                        # else:
                        #     break

                # get the consonant phoneme from the last syllable if necessary
                if self.consonant_dupe and len(sylpron) == 0 and sylidx > 0 \
                        and syl_first_letter != phoneme_first_letter \
                        and syl_first_letter in self.consonants:
                    last_pron = sylprons[syllables[sylidx - 1]]
                    last_phoneme = last_pron[len(last_pron) - 1]
                    if last_phoneme[0].lower() == syl_first_letter:
                        sylpron.append(last_phoneme)

                sylpron.append(phoneme)
                pronidx += 1
            # append the rest to the last syllable if we missed some
            if syllables.index(syl) == len(syllables) - 1 and pronidx < len(pron):
                length = len(pron) - pronidx
                for i in range(pronidx, pronidx + length):
                    sylpron.append(pron[i])
            sylprons[syl] = sylpron
        return sylprons

    def get_token_phonemes(self, token: Token) -> dict[str]:

        phonemes = dict[str]()
        phonemes["whole_word"] = None
        phonemes["syllables"] = None

        # if pronounce acronyms
        if self.sound_out_acronyms and self.is_acronym(token.text):
            word = '.'.join([ch for ch in self.normalize_word(token.text)])
            word += '.'
            if self.syllable_grouping:
                acr_syls = [ch for ch in token.text.strip('.')]
                token._.set("syllables", acr_syls)
                token._.set("syllables_count", len(acr_syls))

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

        tag = token.tag_

        phonemes["whole_word"] = self.g2p.get_word_phonemes(word, tag)
        if self.syllable_grouping or token.pos_ == "CONTRACTION":
            if not token._.syllables:
                phonemes["syllables"] = dict[str]()
                phonemes["syllables"][token.text.lower()] = phonemes["whole_word"]
            else:
                phonemes["syllables"] = self.group_syllables("", phonemes["whole_word"], token._.syllables)
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
                    and len(doc[tokenIdx + 1].text) > 1 \
                    and "'" in doc[tokenIdx + 1].text:
                skip = self.handle_contraction(tokenIdx, doc)
                if skip > 0:
                    tokenIdx += skip
                    continue
                # If not a valid contraction, keep going
            self.set_syllables(token)
            phonemes = self.get_token_phonemes(token)
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
        tokens: list[Token] = [doc[tokenIdx]]
        idx = tokenIdx
        while idx < len(doc) - 1:
            idx += 1
            next_token = doc[idx]
            if "'" in next_token.text:
                tokens.append(next_token)
            else:
                break
        whole_word = self.normalize_word("".join([t.text for t in tokens]))
        if whole_word in self.contractions:
            contraction_phones = dict[str]()
            syllables = [t.text for t in tokens]
            contraction_phones["whole_word"] = self.g2p.get_word_phonemes(whole_word)
            contraction_phones["syllables"] = self.group_syllables("", contraction_phones["whole_word"], syllables)
            t: Token
            for t in tokens:
                t_phone = dict()
                t_phone["whole_word"] = contraction_phones["syllables"][t.text]
                if self.syllable_grouping:
                    t_phone["syllables"] = dict()
                    t_phone["syllables"][t.text] = contraction_phones["syllables"][t.text]
                    if not t._.syllables:
                        t._.set("syllables", [t.text])
                        t._.set("syllables_count", 1)
                t._.set("phonemes", t_phone)
            return len(tokens)
        return 0
