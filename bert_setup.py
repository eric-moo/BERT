from datasets import *
from transformers import *
from tokenizers import *
import os
import json

dataset = load_dataset('onestop_english', 'default', split="train")

# split the dataset into training (90%) and testing (10%)
d = dataset.train_test_split(test_size=0.1)

def dataset_to_text(dataset, output_filename="data.txt"):
  with open(output_filename, "w",encoding="utf-8") as f:
    for t in dataset["text"]:
      print(t, file=f)

# save the training set to train.txt
dataset_to_text(d["train"], "train.txt")
# save the testing set to test.txt
dataset_to_text(d["test"], "test.txt")

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["train.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 256
# whether to truncate
truncate_longer_samples = True

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
# train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

model_path = "bert_pretrain"
# make the directory if not already there
if not os.path.isdir(model_path):
  os.mkdir(model_path)
# save the tokenizer  
tokenizer.save_model(model_path)
# dumping some of the tokenizer config to config file, 
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)

dataset = load_dataset('onestop_english', 'default', split="train")
d = dataset.train_test_split(test_size=0.5)
model_path = "bert-base-uncased"
special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["train.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
# whether to truncate
truncate_longer_samples = True

tokenizer = BertTokenizerFast.from_pretrained(model_path)

def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as 
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

train_dataset, test_dataset

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True, batch_size=2_000,
                                    desc=f"Grouping texts in chunks of {max_length}")
  test_dataset = test_dataset.map(group_texts, batched=True, batch_size=2_000,
                                  num_proc=4, desc=f"Grouping texts in chunks of {max_length}")

model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM.from_pretrained(model_path,config=model_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

training_args = TrainingArguments(
    output_dir="berttt",          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=4,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=4, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=32,  # evaluation batch size
    logging_steps=50,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=50,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
