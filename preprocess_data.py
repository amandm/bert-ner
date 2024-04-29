import json
from transformers import BertJapaneseTokenizer
from label import label2id

MAX_LENGTH = 128  # Maximum token count per sentence
BERT_MODEL = "cl-tohoku/bert-base-japanese-v2"  # Pretrained model to be used
DATASET_PATH = "ner.json"
TAGGED_DATASET_PATH = "ner_tagged.json"

# 1. Data Loading

with open(DATASET_PATH) as f:
  ner_data_list = json.load(f)

# 2. Named Entity Tagging

# Adjusts the start and end positions of entities due to spaces affecting tokenization
def adjust_entity_span(text, entities):
  white_spece_posisions = [i for i, c in enumerate(text) if c == " "]
  for entity in entities:
    start_pos = entity["span"][0]
    end_pos = entity["span"][1]
    start_diff = sum(white_spece_pos < start_pos for white_spece_pos in white_spece_posisions)
    end_diff = sum(white_spece_pos < end_pos for white_spece_pos in white_spece_posisions)
    entity["span"] = [start_pos - start_diff, end_pos - end_diff]

for ner_data in ner_data_list:
  adjust_entity_span(ner_data["text"], ner_data["entities"])

sentence_list = [ner_data["text"] for ner_data in ner_data_list]

tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)

encoded_sentence_list = [tokenizer(sentence, max_length=MAX_LENGTH, padding="max_length", truncation=True) for sentence in sentence_list]

def calc_token_length(token):
  return len(token) -2 if token.startswith("##") else len(token)

def warn_start_pos(pos, token, entity, curid):
  print("[WARN] Token start position exceeds entity start position. Entity start=<" + str(entity["span"][0]) + "> Token start=<" + str(pos) + "> curid=<" + curid + "> token=<" + token + "> entity=<" + entity["name"] + ">")

def warn_end_pos(pos, token, entity, curid):
  token_length = calc_token_length(token)
  print("[WARN] Token end position exceeds entity end position. Entity end=<" + str(entity["span"][1]) + "> Token end=<" + str(pos + token_length) + "> curid=<" + curid + "> token=<" + token + "> entity=<" + entity["name"] + ">")

def search_tokens(tokens, entity, curid):
  ret = {}

  entity_type = entity["type"]
  entity_span = entity["span"]
  entity_start_pos = entity_span[0]
  entity_end_pos = entity_span[1]

  pos = 0
  is_inside_entity = False
  for i, token in enumerate(tokens):
    if token in ["[UNK]", "[SEP]", "[PAD]"]:
      break
    elif token == "[CLS]":
      continue

    token_length = calc_token_length(token)
    if not is_inside_entity:
      if pos == entity_start_pos:
        ret[i] = "B-" + entity_type
        if pos + token_length == entity_end_pos:
          break
        elif pos + token_length < entity_end_pos:
          is_inside_entity = True
        else:
          warn_end_pos(pos, token, entity, curid)
          print(tokens)
      elif pos > entity_start_pos:
        warn_start_pos(pos, token, entity, curid)
        print(tokens)
        break
    else:
      if pos + token_length == entity_end_pos:
        ret[i] = "I-" + entity_type
        is_inside_entity = False
        break
      elif pos + token_length < entity_end_pos:
        ret[i] = "I-" + entity_type
      else:
        warn_end_pos(pos, token, entity, curid)
        print(tokens)
        ret.clear()
        is_inside_entity = False
        break
    pos += token_length

  return ret

# Tagging tokens
tags_list = []
for i, encoded_sentence in enumerate(encoded_sentence_list):
  tokens = tokenizer.convert_ids_to_tokens(encoded_sentence["input_ids"])

  tags = ["O"] * MAX_LENGTH

  ner_data = ner_data_list[i]
  curid = ner_data["curid"]

  entities = ner_data["entities"]

  for entity in entities:
    found_token_pos_tags = search_tokens(tokens, entity, curid)
    for pos, tag in found_token_pos_tags.items():
      tags[pos] = tag

  tags_list.append(tags)

# Convert named entity tags to IDs for training use
encoded_tags_list = [[label2id[tag] for tag in tags] for tags in tags_list]

# Save tagged data
tagged_sentence_list = []
for encoded_sentence, encoded_tags in zip(encoded_sentence_list, encoded_tags_list):
  tagged_sentence = {}
  tagged_sentence['input_ids'] = encoded_sentence['input_ids']
  tagged_sentence['token_type_ids'] = encoded_sentence['token_type_ids']
  tagged_sentence['attention_mask'] = encoded_sentence['attention_mask']
  tagged_sentence['labels'] = encoded_tags
  tagged_sentence_list.append(tagged_sentence)

with open(TAGGED_DATASET_PATH, 'w') as f:
  json.dump(tagged_sentence_list, f)
