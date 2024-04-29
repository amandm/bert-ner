# タグ一覧
id2label = {
  0: "O",
  1: "B-人名",
  2: "I-人名",
  3: "B-法人名",
  4: "I-法人名",
  5: "B-政治的組織名",
  6: "I-政治的組織名",
  7: "B-その他の組織名",
  8: "I-その他の組織名",
  9: "B-地名",
  10: "I-地名",
  11: "B-施設名",
  12: "I-施設名",
  13: "B-製品名",
  14: "I-製品名",
  15: "B-イベント名",
  16: "I-イベント名"
}

label2id = {label: id for id, label in id2label.items()}

# import json
# DATAPATH = "ner.json"
# def create_labels(path):
#     label2id = {"O": 0}
#     json_dict = json.load(open(path, "r"))
#     entity_types = []
#     for unit in json_dict:
#         for entity in unit["entities"]:
#             entity_types.append(entity["type"])
#     entity_types = set(sorted(entity_types))
#     label2id = {"O": 0}
#     for i, entity_type in enumerate(entity_types):
#         # "B-"
#         label2id[f"B-{entity_type}"] = i * 2 + 1
#         # "I-"
#         label2id[f"I-{entity_type}"] = i * 2 + 2
#     return label2id

# label2id = create_labels(DATAPATH)
# id2label = {v: k for k, v in label2id.items()}

# {0: 'O',
#  1: 'B-その他の組織名',
#  2: 'I-その他の組織名',
#  3: 'B-イベント名',
#  4: 'I-イベント名',
#  5: 'B-人名',
#  6: 'I-人名',
#  7: 'B-地名',
#  8: 'I-地名',
#  9: 'B-政治的組織名',
#  10: 'I-政治的組織名',
#  11: 'B-施設名',
#  12: 'I-施設名',
#  13: 'B-法人名',
#  14: 'I-法人名',
#  15: 'B-製品名',
#  16: 'I-製品名'}
