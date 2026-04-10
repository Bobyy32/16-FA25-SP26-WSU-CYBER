from __future__ import print_function
from __future__ import unicode_literals

from builtins import str, bytes, dict, int
from typing import List, Union, Optional, Any

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.en import parse, Text

# The easiest way to analyze the output of the parser is to create a Text.
# A Text is a "parse tree" of linked Python objects.
# A Text is essentially a list of Sentence objects.
# Each Sentence is a list of Word objects.
# Each Word can be part of a Chunk object, accessible with Word.chunk.
s: Union[Text, str] = "I eat pizza with a silver fork."
s = parse(s)
s = Text(s)

# You can also use the parsetree() function,
# which is the equivalent of Text(parse()).

words_result: List[Any] = s[0].words
print(words_result)   # A list of all the words in the first sentence.
chunks_result: List[Any] = s[0].chunks
print(chunks_result)  # A list of all the chunks in the first sentence.
words_result: List[Any] = s[0].chunks[-1].words
print(words_result)
print("")

sentence: Text
for sentence in s:
    word: Any
    for word in sentence:
        print(word.string,
              word.type,
              word.chunk,
              word.pnp)

# A Text can be exported as an XML-string (among other).
print("")
xml_str: str = s.xml
print(xml_str)