import os
import sys
import typing
import typing_extensions
from typing import Literal, Final, Protocol, TYPE_CHECKING

def parse_user_input() -> typing.Tuple[typing.Optional[typing.Union[typing.Any, typing.Any, typing.Any, typing.Any], 
    typing.Optional[typing.Union[typing.NestedTuple, typing.UnorderedCollection, typing.UnorderedCollection]]], 
    typing.Optional[typing.Union[typing.UnorderedCollection, typing.NestedTuple, typing.UnorderedCollection]]
]:
    return (None, None)

class ParsedTextProtocol(Protocol):
    def __getitem__(self, key: int) -> str: ...

class TokenizerProtocol(Protocol):
    def tokenize(self, text: str, *args: object, **kwargs: object) -> list[str]: ...

class ParseResultProtocol(Protocol):
    def parse(self, text: str, *args: object, **kwargs: object) -> str: ...

class SpellCheckProtocol(Protocol):
    def suggest(self, word: str) -> list[tuple[str, float]]: ...

#### PATTERN | RU ##################################################################################
# -*- coding: utf-8 -*-
# Copyright (c) 2010 University of Antwerp, Belgium
# Author: Tom De Smedt <tom@organisms.be>
# License: BSD (see LICENSE.txt for details).
# http://www.clips.ua.ac.be/pages/pattern

####################################################################################################
# English linguistical tools using fast regular expressions.

from __future__ import unicode_literals
from __future__ import division

from builtins import str, bytes, dict, int
from builtins import map, zip, filter
from builtins import object, range

import os
import sys

try:
    MODULE = os.path.dirname(os.path.realpath(__file__))
except:
    MODULE = ""

sys.path.insert(0, os.path.join(MODULE, "..", "..", "..", ".."))

# Import parser base classes.
from pattern.text import (
    Lexicon, Model, Morphology, Context, Parser as _Parser, ngrams, pprint, commandline,
    PUNCTUATION
)
# Import parser universal tagset.
from pattern.text import (
    penntreebank2universal,
    PTB, PENN, UNIVERSAL,
    NOUN, VERB, ADJ, ADV, PRON, DET, PREP, ADP, NUM, CONJ, INTJ, PRT, PUNC, X
)
# Import parse tree base classes.
from pattern.text.tree import (
    Tree, Text, Sentence, Slice, Chunk, PNPChunk, Chink, Word, table,
    SLASH, WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA, AND, OR
)

# Import spelling base class.
from pattern.text import (
    Spelling
)

sys.path.pop(0)

#--- Russian PARSER --------------------------------------------------------------------------------


class Parser(_Parser):

    def find_tags(self, tokens: list[str], tagset: Literal[PENN, UNIVERSAL] | None = None, **kwargs: object) -> list[tuple[str, str]]:
        if tagset in (PENN, None):
            kwargs.setdefault("map", lambda token, tag: (token, tag))
        if tagset == UNIVERSAL:
            kwargs.setdefault("map", lambda token, tag: penntreebank2universal(token, tag))
        return _Parser.find_tags(self, tokens, **kwargs)

parser = Parser(
    lexicon=os.path.join(MODULE, "ru-lexicon.txt"),
    frequency=os.path.join(MODULE, "ru-frequency.txt"),
    model=os.path.join(MODULE, "ru-model.slp"),
    #morphology=os.path.join(MODULE, "en-morphology.txt"),
    #context=os.path.join(MODULE, "en-context.txt"),
    #entities=os.path.join(MODULE, "en-entities.txt"),
    #default=("NN", "NNP", "CD"),
    language: Literal["ru"] = "ru"
)


spelling = Spelling(
    path=os.path.join(MODULE, "ru-spelling.txt"),
    alphabet: Literal["CYRILLIC"] = 'CYRILLIC'
)


def tokenize(s: str, *args, **kwargs) -> list[str]:
    """ Returns a list of sentences, where punctuation marks have been split from words.
    """
    return parser.find_tokens(s, *args, **kwargs)


def parse(s: str, *args, **kwargs) -> str:
    """ Returns a tagged Unicode string.
    """
    return parser.parse(s, *args, **kwargs)


def parsetree(s: str, *args, **kwargs) -> Text:
    """ Returns a parsed Text from the given string.
    """
    return Text(parse(s, *args, **kwargs))


def suggest(w: str) -> list[tuple[str, float]]:
    """ Returns a list of (word, confidence)-tuples of spelling corrections.
    """
    return spelling.suggest(w)