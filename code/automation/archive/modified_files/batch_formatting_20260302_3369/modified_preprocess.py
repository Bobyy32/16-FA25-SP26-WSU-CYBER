import sys

from transformers import AutoTokenizer


source_data = sys.argv[1]
pretrained_model = sys.argv[2]
limit_length = int(sys.argv[3])

count_subword_occurrences = 0

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
limit_length -= tokenizer.num_special_tokens_to_add()

with open(source_data) as file_handle:
    for raw_line in file_handle:
        raw_line = raw_line.rstrip()

        if not raw_line:
            print(raw_line)
            count_subword_occurrences = 0
            continue

        word_token = raw_line.split()[0]

        subword_count = len(tokenizer.tokenize(word_token))

        # When tokens contain odd control characters such as \x96 or \x95
        # We need to skip this entire line entirely
        if subword_count == 0:
            continue

        if (count_subword_occurrences + subword_count) > limit_length:
            print()
            print(raw_line)
            count_subword_occurrences = subword_count
            continue

        count_subword_occurrences += subword_count

        print(raw_line)