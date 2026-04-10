from __future__ import print_function
from __future__ import unicode_literals

import pattern.en

parse = pattern.en.parse
Text = pattern.en.Text

s = "I eat pizza with a silver fork."
s = parse(s)
s = Text(s)

print(s[0].words)   # A list of all the words in the first sentence.
print(s[0].chunks)  # A list of all the chunks in the first sentence.
print(s[0].chunks[-1].words)
print("")

for sentence in s:
    for word in sentence:
        print(word.string,
              word.type,
              word.chunk,
              word.pnp)

# A Text can be exported as an XML-string (among other).
print("")
print(s.xml)