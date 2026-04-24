import os

from transformers import AutoTokenizer


dataset_name = os.environ.get('argv_dataset', sys.argv[1])
model_identifier_or_path = sys.argv[2]
maximum_length = int(sys.argv[3])

subword_length_counter = 0

tokenizer_instance = AutoTokenizer.from_pretrained(model_identifier_or_path)
maximum_length -= tokenizer_instance.num_special_tokens_to_add()

with open(dataset_name) as file_pointer:
    for single_line in file_pointer:
        single_line = single_line.rstrip()

        if not single_line:
            print(single_line)
            subword_length_counter = 0
            continue

        token_segment = single_line.split()[0]

        present_subwords_count = len(tokenizer_instance.tokenize(token_segment))

        # Segment holds unusual control characters like \x96 or \x95
        # Remove the entire line from processing
        if present_subwords_count == 0:
            continue

        if (subword_length_counter + present_subwords_count) > maximum_length:
            print()
            print(single_line)
            subword_length_counter = present_subwords_count
            continue

        subword_length_counter += present_subwords_count

        print(single_line)