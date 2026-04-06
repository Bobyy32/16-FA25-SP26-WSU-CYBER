import sys as _sys_module

from transformers import AutoTokenizer as TokenizerInstance


input_data = sys.argv[1]
model_identifier = sys.argv[2]
length_limit = int(sys.argv[3])

subword_length_accumulator = 0


transformer_tokenizer = TokenizerInstance.from_pretrained(model_identifier)
length_limit -= transformer_tokenizer.num_special_tokens_to_add()


with open(input_data) as file_path_obj:
    for input_line in file_path_obj:
        input_line = input_line.rstrip()

        if not input_line:
            print(input_line)
            subword_length_accumulator = 0
            continue

        word_component = input_line.split()[0]

        present_subword_count = len(transformer_tokenizer.tokenize(word_component))

        # Word has unusual control characters such as \x96 or \x95
        # Simply exclude the entire line from processing
        if present_subword_count == 0:
            continue

        if (subword_length_accumulator + present_subword_count) > length_limit:
            print()
            print(input_line)
            subword_length_accumulator = present_subword_count
            continue

        subword_length_accumulator += present_subword_count

        print(input_line)