This script is a complete training pipeline for fine-tuning an XLNet language model on a text dataset using Hugging Face's `transformers` library and `datasets` library. It's designed to work with permutation language modeling (PLM), which is a training technique used in models like XLNet where the model learns to predict tokens in a permuted order.

Here's a breakdown of the key components:

### 1. **Imports and Setup**
- Imports necessary libraries (`transformers`, `datasets`, `torch`, etc.)
- Sets up logging and argument parsing

### 2. **Model and Tokenizer Loading**
- Loads configuration and tokenizer from a pre-trained XLNet model or creates a new one
- Loads the XLNet language model (`XLNetLMHeadModel`)
- Resizes embeddings if needed to accommodate new tokens

### 3. **Data Preprocessing**
- Tokenizes the dataset
- Handles both line-by-line tokenization and grouped text tokenization
- Uses `DataCollatorForPermutationLanguageModeling` to prepare batches for permutation language modeling

### 4. **Training Setup**
- Initializes the `Trainer` with:
  - Model
  - Training arguments
  - Datasets
  - Tokenizer
  - Data collator

### 5. **Training and Evaluation**
- Trains the model
- Evaluates the model
- Saves results and metrics

### Key Features:
- **Permutation Language Modeling**: The script uses a specialized data collator for PLM training
- **Flexible Data Handling**: Supports both line-by-line and grouped text processing
- **Checkpointing**: Supports resuming training from checkpoints
- **Evaluation Metrics**: Computes perplexity and other metrics
- **Push to Hub**: Option to push trained models to Hugging Face Model Hub

### Usage:
```bash
python train_xlnet.py \
    --model_name_or_path xlnet-base-cased \
    --dataset_name wikipedia \
    --output_dir ./xlnet-wiki \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100
```

This script provides a complete, production-ready solution for training XLNet models on custom datasets with permutation language modeling.