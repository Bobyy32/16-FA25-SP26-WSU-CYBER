### **Key Parameters:**

- `--model_name_or_path`: Path or name of the pretrained model.
- `--train_file`: Path to the training text file.
- `--output_dir`: Directory to save the trained model.
- `--do_train`, `--do_eval`: Flags to enable training and evaluation.
- `--per_device_train_batch_size`, `--per_device_eval_batch_size`: Batch sizes for training and evaluation.
- `--num_train_epochs`: Number of training epochs.
- `--save_steps`, `--logging_steps`: Frequency of saving and logging.

### **Important Notes:**

- The script uses permutation language modeling (PLM) by default, which is a variant of masked language modeling.
- It supports distributed training and TPU training via `xla_spawn`.
- The model is saved in the `output_dir` with tokenizer and configuration files.

### **Dependencies:**

Ensure you have the following installed: