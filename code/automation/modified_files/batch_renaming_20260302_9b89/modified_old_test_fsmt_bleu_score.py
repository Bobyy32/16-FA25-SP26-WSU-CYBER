# Copyright 2020 Huggingface
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest

from parameterized import parameterized

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers.testing_utils import get_tests_dir, require_torch, slow, torch_device
from utils import calculate_bleu


file_path = get_tests_dir() + "/test_data/fsmt/fsmt_val_data.json"
with open(file_path, encoding="utf-8") as data_file:
    evaluation_data = json.load(data_file)


@require_torch
class ModelEvalTester(unittest.TestCase):
    def fetch_tokenizer(self, model_name):
        return FSMTTokenizer.from_pretrained(model_name)

    def fetch_model(self, model_name):
        model_instance = FSMTForConditionalGeneration.from_pretrained(model_name).to(torch_device)
        if torch_device == "cuda":
            model_instance.half()
        return model_instance

    @parameterized.expand(
        [
            ["en-ru", 26.0],
            ["ru-en", 22.0],
            ["en-de", 22.0],
            ["de-en", 29.0],
        ]
    )
    @slow
    def test_bleu_scores(self, language_pair, minimum_score):
        # note: this test is not testing the best performance since it only evals a small batch
        # but it should be enough to detect a regression in the output quality
        model_identifier = f"facebook/wmt19-{language_pair}"
        tokenizer_instance = self.fetch_tokenizer(model_identifier)
        model = self.fetch_model(model_identifier)

        source_sentences = evaluation_data[language_pair]["src"]
        target_sentences = evaluation_data[language_pair]["tgt"]

        input_batch = tokenizer_instance(source_sentences, return_tensors="pt", truncation=True, padding="longest").to(torch_device)
        generated_outputs = model.generate(
            input_ids=input_batch.input_ids,
            num_beams=8,
        )
        decoded_texts = tokenizer_instance.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        bleu_metrics = calculate_bleu(decoded_texts, target_sentences)
        print(bleu_metrics)
        self.assertGreaterEqual(bleu_metrics["bleu"], minimum_score)