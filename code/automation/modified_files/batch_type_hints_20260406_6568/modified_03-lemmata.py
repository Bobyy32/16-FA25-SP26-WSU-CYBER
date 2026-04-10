from __future__ import print_function
from __future__ import unicode_literals
from builtins import str, bytes, dict, int

import os
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.search import search, match
from pattern.en import parsetree

def parse_sentence_input() -> typing.Tuple[str, typing.Optional[str]]:
    """
    Parse and validate input sentence for processing.
    This function handles parsing sentence inputs for grammatical analysis.
    """
    s: str = ""
    t: typing.Optional[parsetree] = None
    m: typing.Optional[match] = None
    
    sentences: typing.List[str] = [
        "the turtle was faster than the hare",
        "Arnold Schwarzenegger is more dangerous than Dolph Lundgren"
    ]
    
    p: str = "NP be ADJP|ADVP than NP"
    
    for sentence in sentences:
        t = parsetree(sentence, lemmata=True)  # parse lemmas
        
        m = search(p, t)
        
        if m:
            # Constituents for the given constraint indices:
            # 0 = NP, 2 = ADJP|ADVP, 4 = NP
            constituents_result = m[0].constituents(constraint=[0, 2, 4])
            
            print(constituents_result)
            print("")
    
    return s, str(t) if t else None

def parse_sentence_input_single() -> typing.Tuple[str, str, typing.Optional[typing.List[typing.Tuple[str, str]]]]:
    """
    Parse a single sentence and extract word constraints.
    """
    p: str = "NP be ADJP|ADVP than NP"
    t: parsetree = parsetree("the turtle was faster than the hare", lemmata=True)
    m: typing.Optional[match] = match(p, t)
    
    words_result: typing.Optional[typing.List[typing.Tuple[str, str]]] = None
    
    if m:
        words_result = []
        for word in m.words:
            constraint_result = m.constraint(word)
            words_result.append((word, constraint_result))
        
        print(t)
        print("")
        
        for word_tuple in words_result:
            word: str = word_tuple[0]
            constraint: str = word_tuple[1]
            print("%s\t=> %s" % (word, constraint))
    
    return str(t), str(m), words_result

def generate_constituents_result() -> typing.Optional[str]:
    """
    Generate constituents result for the given constraint indices.
    """
    s: typing.Optional[str] = None
    t: typing.Optional[parsetree] = None
    
    if t is not None:
        # Generate constituents result
        return str(t)
    else:
        # Return None if t is empty
        return None