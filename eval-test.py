from transformers import pipeline
import numpy as np
import torch
import json
from label import label2id, id2label 
from transformers import BertJapaneseTokenizer, BertForTokenClassification

MODEL_DIR = "./model_v2/checkpoint-18000/"
MAX_SEQUENCE_LENGTH = 128  
DATASET_PATH = "ner_tagged.json"
MODEL_SAVE_DIR = "./model_v2"
LOGGING_DIR = "./logs"

model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR,model_max_length=128)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Load the test dataset
with open(DATASET_PATH, 'r') as file:
    tagged_sentences = json.load(file)

from sklearn.model_selection import train_test_split
_, test_sentences = train_test_split(tagged_sentences, test_size=0.2, random_state=126)  # Assuming 20% as test split

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, tagged_sentences):
        self.tagged_sentences = tagged_sentences

    def __len__(self):
        return len(self.tagged_sentences)

    def __getitem__(self, index):
        item = {key: torch.tensor(value).to(device) for key, value in self.tagged_sentences[index].items()}
        return item

test_dataset = NERDataset(test_sentences)

from transformers import Trainer, TrainingArguments


# Load the metrics
import evaluate
ner_metric = evaluate.load("seqeval")

def compute_metrics(prediction_output):
    predictions, labels = prediction_output.predictions, prediction_output.label_ids
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = ner_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Evaluation setup
training_arguments = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=32
)

eval_trainer = Trainer(
    model=model,
    args=training_arguments,
    compute_metrics=compute_metrics,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Run evaluation
results = eval_trainer.evaluate()
print(results)

examples = [
    "株式会社はJurabi、東京都台東区に本社を置くIT企業である","レッドフォックス株式会社は、東京都千代田区に本社を置くITサービス企業である"
]
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
for text in examples:
    entities = ner_pipeline(text)
    print(entities)
    print("-"*70)
    

