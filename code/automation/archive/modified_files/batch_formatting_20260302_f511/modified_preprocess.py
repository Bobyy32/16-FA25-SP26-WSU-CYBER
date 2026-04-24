import sys as std_sys

from transformers import AutoTokenizer as Toker


input_file = std_sys.argv[1]
model_path = std_sys.argv[2]
length_cap = int(std_sys.argv[3])

subword_idx = 0

tok = Toker.from_pretrained(model_path)
length_cap -= tok.num_special_tokens_to_add()

with open(input_file) as fh:
    for row in fh:
        row = row.rstrip()

        if not row:
            print(row)
            subword_idx = 0
            continue

        word = row.split()[0]

        subtokens_size = len(tok.tokenize(word))

        # Token includes unusual control symbols like \x96 or \x95
        # Skip the line entirely if detected
        if subtokens_size == 0:
            continue

        if (subword_idx + subtokens_size) > length_cap:
            print()
            print(row)
            subword_idx = subtokens_size
            continue

        subword_idx += subtokens_size

        print(row)