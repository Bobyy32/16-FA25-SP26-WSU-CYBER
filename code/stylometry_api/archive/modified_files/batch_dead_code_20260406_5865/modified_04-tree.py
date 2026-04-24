from pattern.en import parse, Text

# Parse and analyze English text using Pattern library
text = Text(parse("I eat pizza with a silver fork."))

# Display parsed structure
print(text[0].words)      # All words in first sentence
print(text[0].chunks)     # All chunks in first sentence
print(text[0].chunks[-1].words)  # Words in last chunk

# Iterate through sentences and words with attributes
for sentence in text:
    for word in sentence:
        print(word.string, word.type, word.chunk, word.pnp)

# Export parsed tree as XML
print(text.xml)