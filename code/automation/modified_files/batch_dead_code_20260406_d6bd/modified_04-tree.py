import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.en import parse, Text

s = "I eat pizza with a silver fork."
s = parse(s)
s = Text(s)

print(s[0].words)
print(s[0].chunks)
print(s[0].chunks[-1].words)

for sentence in s:
    for word in sentence:
        print(word.string,
              word.type,
              word.chunk,
              word.pnp)

print(s.xml)