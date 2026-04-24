from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
import sys
from builtins import str, bytes, dict, int
from builtins import map, zip, filter
from builtins import object, range

try:
    MODULE = os.path.dirname(os.path.realpath(__file__))
except Exception:
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


class RussianParserException(Exception):
    """Custom exception for Russian parser errors."""
    pass


class RussianParser(_Parser):
    """A Russian language parser based on Pattern library with enhanced error handling."""

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        frequency_path: Optional[str] = None,
        model_path: Optional[str] = None,
        language: str = "ru"
    ) -> None:
        """Initialize Russian parser with configuration paths and language settings.

        Args:
            lexicon_path: Path to lexicon file (word -> tag mappings).
            frequency_path: Path to word frequency file.
            model_path: Path to trained classifier model file.
            language: Language code (default: "ru" for Russian).

        Raises:
            RussianParserException: If file paths are invalid or parser fails to initialize.
        """
        self.language = language
        self.lexicon_path = lexicon_path or os.path.join(MODULE, "ru-lexicon.txt")
        self.frequency_path = frequency_path or os.path.join(MODULE, "ru-frequency.txt")
        self.model_path = model_path or os.path.join(MODULE, "ru-model.slp")
        
        try:
            super().__init__(
                lexicon=lexicon_path,
                frequency=frequency_path,
                model=model_path,
                language=language
            )
        except Exception as e:
            raise RussianParserException(f"Parser initialization failed: {str(e)}")

    def find_tags(
        self,
        tokens: List[str],
        tagset: Optional[str] = None,
        **kwargs: Any
    ) -> List[Tuple[str, str]]:
        """Find and tag tokens using configured tagset.

        Args:
            tokens: List of tokenized words to tag.
            tagset: Tagset to use (default: PENN or None).
            **kwargs: Additional configuration options.

        Returns:
            List of tuples containing (token, tag).

        Raises:
            RussianParserException: If tagset is invalid.
        """
        if tagset in (PENN, None):
            kwargs.setdefault("map", lambda token, tag: (token, tag))
        elif tagset == UNIVERSAL:
            kwargs.setdefault("map", lambda token, tag: penntreebank2universal(token, tag))
        else:
            raise RussianParserException(f"Invalid tagset: {tagset}")
        
        try:
            return _Parser.find_tags(self, tokens, **kwargs)
        except Exception as e:
            raise RussianParserException(f"Tagging failed: {str(e)}")

    def parse(
        self,
        text: str,
        **kwargs: Any
    ) -> str:
        """Parse a text string and return tagged output.

        Args:
            text: Input text string to parse.
            **kwargs: Additional parsing options.

        Returns:
            Tagged Unicode string.

        Raises:
            RussianParserException: If parsing fails.
        """
        if not text or not isinstance(text, str):
            raise RussianParserException("Input text must be a non-empty string.")
        
        try:
            return _Parser.parse(self, text, **kwargs)
        except Exception as e:
            raise RussianParserException(f"Parsing failed: {str(e)}")


# Singleton parser instance with fallback configuration
parser = RussianParser(
    lexicon=os.path.join(MODULE, "ru-lexicon.txt"),
    frequency=os.path.join(MODULE, "ru-frequency.txt"),
    model=os.path.join(MODULE, "ru-model.slp"),
    language="ru"
)


class RussianSpelling:
    """Spelling correction module for Russian text."""

    def __init__(self, path: str, alphabet: str = 'CYRILLIC') -> None:
        """Initialize spelling module.

        Args:
            path: Path to spelling rules file.
            alphabet: Character set (default: 'CYRILLIC').

        Raises:
            RussianParserException: If file is invalid or alphabet is unsupported.
        """
        self.alphabet = alphabet
        self.spelling_path = path
        
        try:
            self.spelling = Spelling(path=path, alphabet=alphabet)
        except Exception as e:
            raise RussianParserException(f"Spelling initialization failed: {str(e)}")

    def suggest(
        self,
        word: str
    ) -> List[Tuple[str, float]]:
        """Suggest spelling corrections for a word.

        Args:
            word: Input word to suggest corrections for.

        Returns:
            List of (word, confidence)-tuples.

        Raises:
            RussianParserException: If word is empty or suggestion fails.
        """
        if not word:
            raise RussianParserException("Empty word for suggestion.")
        
        try:
            return self.spelling.suggest(word)
        except Exception as e:
            raise RussianParserException(f"Suggestion failed: {str(e)}")


# Convenience functions with enhanced error handling

def tokenize(
    s: str,
    *args: Any,
    **kwargs: Any
) -> List[str]:
    """Tokenize a string and return list of sentences with punctuation separated.

    Args:
        s: Input string to tokenize.
        *args: Additional parser arguments.
        **kwargs: Additional parsing options.

    Returns:
        List of tokenized sentences.

    Raises:
        RussianParserException: If input is empty or tokenization fails.
    """
    if not s or not isinstance(s, str):
        raise RussianParserException("Input must be a non-empty string.")
    
    try:
        return parser.find_tokens(s, *args, **kwargs)
    except Exception as e:
        raise RussianParserException(f"Tokenization failed: {str(e)}")


def parse(
    s: str,
    *args: Any,
    **kwargs: Any
) -> str:
    """Parse a string and return tagged output.

    Args:
        s: Input string to parse.
        *args: Additional parser arguments.
        **kwargs: Additional parsing options.

    Returns:
        Tagged Unicode string.

    Raises:
        RussianParserException: If input is invalid or parsing fails.
    """
    if not s or not isinstance(s, str):
        raise RussianParserException("Input must be a non-empty string.")
    
    try:
        return parser.parse(s, *args, **kwargs)
    except Exception as e:
        raise RussianParserException(f"Parsing failed: {str(e)}")


def parsetree(
    s: str,
    *args: Any,
    **kwargs: Any
) -> Text:
    """Parse a string and return a parse tree.

    Args:
        s: Input string to parse.
        *args: Additional parser arguments.
        **kwargs: Additional parsing options.

    Returns:
        Parse tree object (Text).

    Raises:
        RussianParserException: If input is invalid or parsing fails.
    """
    if not s or not isinstance(s, str):
        raise RussianParserException("Input must be a non-empty string.")
    
    try:
        return Text(parser.parse(s), *args, **kwargs)
    except Exception as e:
        raise RussianParserException(f"Parse tree generation failed: {str(e)}")


def findchunks(
    s: str,
    *args: Any,
    **kwargs: Any
) -> List[Chunk]:
    """Extract chunks from a string.

    Args:
        s: Input string to analyze.
        *args: Additional parser arguments.
        **kwargs: Additional parsing options.

    Returns:
        List of Chunk objects.

    Raises:
        RussianParserException: If input is invalid or extraction fails.
    """
    if not s or not isinstance(s, str):
        raise RussianParserException("Input must be a non-empty string.")
    
    try:
        return parser.findchunks(s, *args, **kwargs)
    except Exception as e:
        raise RussianParserException(f"Chunk extraction failed: {str(e)}")