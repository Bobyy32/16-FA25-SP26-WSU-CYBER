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
    blue_data = json.load(data_file)


@require_torch
class ModelEvalTester(unittest.TestCase):
    def get_tokenizer(self, model_name):
        return FSMTTokenizer.from_pretrained(model_name)

    def get_model(self, model_name):
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
    def test_bleu_scores(self, lang_pair, min_score):
        # note: this test is not testing the best performance since it only evals a small batch
        # but it should be enough to detect a regression in the output quality
        model_name = f"facebook/wmt19-{lang_pair}"
        tokenizer_instance = self.get_tokenizer(model_name)
        model_instance = self.get_model(model_name)

        source_sentences = blue_data[lang_pair]["src"]
        target_sentences = blue_data[lang_pair]["tgt"]

        batch_data = tokenizer_instance(source_sentences, return_tensors="pt", truncation=True, padding="longest").to(torch_device)
        generated_outputs = model_instance.generate(
            input_ids=batch_data.input_ids,
            num_beams=8,
        )
        decoded_results = tokenizer_instance.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        score_results = calculate_bleu(decoded_results, target_sentences)
        print(score_results)
        self.assertGreaterEqual(score_results["bleu"], min_score)