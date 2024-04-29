import json
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification, BertConfig
from label import label2id, id2label  # Importing mapping dictionaries for labels

# Setting logging
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Set the appropriate device based on availability
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Configuration constants
MAX_SEQUENCE_LENGTH = 128  
BERT_MODEL_NAME = "cl-tohoku/bert-base-japanese-v2" 
DATASET_PATH = "ner_tagged.json"
MODEL_SAVE_DIR = "./model_v21"
LOGGING_DIR = "./logs1"

# Load the preprocessed dataset containing tagged sentences for Named Entity Recognition
with open(DATASET_PATH, 'r') as file:
  tagged_sentences = json.load(file)

from sklearn.model_selection import train_test_split

# Define a PyTorch Dataset class to handle the NER data
class NERDataset(torch.utils.data.Dataset):
  def __init__(self, tagged_sentences):
    self.tagged_sentences = tagged_sentences

  def __len__(self):
    return len(self.tagged_sentences)

  def __getitem__(self, index):
    item = {key: torch.tensor(value).to(device) for key, value in self.tagged_sentences[index].items()}
    return item

# Split the data into training and evaluation datasets
training_sentences, evaluation_sentences = train_test_split(tagged_sentences, test_size=0.1,random_state = 126)
training_sentences, test_sentences = train_test_split(training_sentences, test_size=0.1,random_state = 126)

print("train set len = ",len(training_sentences))
print("val set len = ",len(evaluation_sentences))
print("test set len = ",len(test_sentences))

train_dataset = NERDataset(training_sentences)
eval_dataset = NERDataset(evaluation_sentences)
test_dataset = NERDataset(test_sentences)

from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import load_metric

# Configure the model and tokenizer
model_config = BertConfig.from_pretrained(BERT_MODEL_NAME, id2label=id2label, label2id=label2id)
ner_model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, config=model_config).to(device)
tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL_NAME)

# Setup the training arguments
training_arguments = TrainingArguments(
    output_dir = MODEL_SAVE_DIR,
    num_train_epochs = 40,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 32,
    warmup_steps = 500,  # Ramp up from zero to this many steps
    weight_decay = 0.01,  # Weight decay rate
    logging_dir = LOGGING_DIR,
    logging_strategy = "steps",
    logging_steps = 50,
    evaluation_strategy = "steps",
    eval_steps = 500,
    load_best_model_at_end = True,
    metric_for_best_model = "f1",
)


# training_arguments = TrainingArguments(
#     output_dir = MODEL_SAVE_DIR,
#     num_train_epochs = 40,
#     per_device_train_batch_size = 8,
#     per_device_eval_batch_size = 32,
#     warmup_steps = 500, 
#     weight_decay = 0.01, 
#     logging_dir = LOGGING_DIR,
# )

import evaluate
ner_metric = evaluate.load("seqeval")

# Function to compute metrics for the model
def compute_metrics(prediction_output):
    predictions, labels = prediction_output
    predictions = np.argmax(predictions, axis=2)

    # Convert label IDs back to their string representations for metric calculation
    predictions = [
        [id2label[pred] for pred in prediction] for prediction in predictions
    ]
    labels = [
        [id2label[label] for label in label] for label in labels
    ]

    metric_results = ner_metric.compute(predictions=predictions, references=labels)
    print(metric_results)
    return {
        "precision": metric_results["overall_precision"],
        "recall": metric_results["overall_recall"],
        "f1": metric_results["overall_f1"],
        "accuracy": metric_results["overall_accuracy"],
    }

# Initialize and run the trainer
ner_trainer = Trainer(
    model = ner_model, 
    args = training_arguments, 
    compute_metrics = compute_metrics, 
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    tokenizer = tokenizer
)

ner_trainer.train()  # Start the training
ner_trainer.evaluate()  # Evaluate the model

ner_trainer.save_model(MODEL_SAVE_DIR)  # Save the trained model
