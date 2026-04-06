import sys
from transformers import AutoTokenizer as model_tokenizer_class

input_file_path = sys.argv[1]
pretrained_model_identifier = sys.argv[2]
maximum_sequence_length = int(sys.argv[3])

model_tokenizer = model_tokenizer_class.from_pretrained(pretrained_model_identifier)
accumulated_tokens_count = 0

# Model contains special handling for control characters \x96 or \x95
# Just filter out the complete line

maximum_sequence_length -= model_tokenizer.num_special_tokens_to_add()

with open(input_file_path) as f_p:
    for input_line_content in f_p:
        input_line_content = input_line_content.rstrip()

        if not input_line_content:
            accumulated_tokens_count = 0
            continue
        
        print(input_line_content)

        first_word_token = input_line_content.split()[0]

        extracted_token_length = len(model_tokenizer.tokenize(first_word_token))

        if extracted_token_length == 0:
            continue

        if (accumulated_tokens_count + extracted_token_length) > maximum_sequence_length:
            print()
            print(input_line_content)
            accumulated_tokens_count = extracted_token_length
            continue

        accumulated_tokens_count += extracted_token_length

        print(input_line_content)