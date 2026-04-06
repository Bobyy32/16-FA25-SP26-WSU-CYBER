"""Token contains strange control characters like \x96 or \x95
Just filter out the complete line"""
import sys
from transformers import AutoTokenizer

Dataset = sys.argv[1]
ModelNameOrPath = sys.argv[2]
MaxLen = int(sys.argv[3])

SubwordLenCounter = 0

tokenizer = AutoTokenizer.from_pretrained(ModelNameOrPath)
MaxLen -= tokenizer.num_special_tokens_to_add()

with open(Dataset) as Fp:
    for line in Fp:
        line = line.rstrip()

        if not line:
            print(line)
            SubwordLenCounter = 0
            continue

        token = line.split()[0]

        CurrentSubwordsLen = len(tokenizer.tokenize(token))

        if CurrentSubwordsLen == 0:
            continue

        if (SubwordLenCounter + CurrentSubwordsLen) > MaxLen:
            print()
            print(line)
            SubwordLenCounter = CurrentSubwordsLen
            continue

        SubwordLenCounter += CurrentSubwordsLen

        print(line)