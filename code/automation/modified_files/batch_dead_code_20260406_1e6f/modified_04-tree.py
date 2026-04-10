from pattern.en import parse, Text

s = "I eat pizza with a silver fork."
s = parse(s)
s = Text(s)

print(s[0].words)
print(s[0].chunks)
print(s[0].chunks[-1].words)
print("")

for sentence in s:
    for word in sentence:
        print(word.string,
              word.type,
              word.chunk,
              word.pnp)

print("")
print(s.xml)