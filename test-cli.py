# from transformers import pipeline
# import gradio as gr

# from transformers import BertJapaneseTokenizer, BertForTokenClassification

# MODEL_DIR = "./model_v2/checkpoint-18000/"
# model = BertForTokenClassification.from_pretrained(MODEL_DIR)
# tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR,model_max_length=128)
# ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)


# examples = [
#     "株式会社はJurabi、東京都台東区に本社を置くIT企業である",
# ]

# def ner(text):
#     op = ner_pipeline(text)
#     for i in op:
#     # print(i)
#         i['start'] = i["index"]
#         i['end'] = i["index"]
#         print(i)
        
#     styled_html = ""
#     for entity in op:
#         if '法人名' in entity['entity']:
#             color = "lightblue"  # Set your desired color for this entity type
#         elif '地名' in entity['entity']:
#             color = "lightgreen"  # Set your desired color for this entity type
#         else:
#             color = "transparent"  # Default no highlighting
#         entity_html = f'<mark style="background-color: {color}; padding: 0.2em;">{entity["word"]}</mark>'
#         styled_html += entity_html

#     print(styled_html)
#     return styled_html
    
    
#     # return {"text": text, "entities": op}     
    
# demo = gr.Interface(
#             fn = ner,
#              inputs=gr.Textbox(placeholder="Enter sentence here..."), 
#              outputs="html",
#              examples=examples)

# demo.launch()



from transformers import pipeline
from transformers import BertJapaneseTokenizer, BertForTokenClassification

MODEL_DIR = "./model_v2/checkpoint-18000/"

model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR,model_max_length=128)

ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

exp = "株式会社はJurabi、東京都台東区に本社を置くIT企業である"
for i in range(len(exp)):
    print(exp[i],i)
print(exp)
op =ner_pipeline(exp)
for i in op:
    print(i)

# op =ner_pipeline("SPRiNGSと最も仲の良いライバルグループ。")


tokens = tokenizer(exp)
# print(tokens)
decoded = tokenizer.convert_ids_to_tokens(tokens['input_ids'])
print(decoded)

def merge_entities(entities, tokens, sentence):
    current_entity = None
    merged_entities = []
    token_positions = []
    last_index = 0

    # Calculate start positions of each token in the sentence
    for token in tokens:
        if token == '[CLS]' or token == '[SEP]':
            # print("Special token")
            start = -1
            end = 0
        
        elif token[:2] == "##":
            start = sentence.find(token[2:], last_index)
            end = start + len(token[2:])
        else:
            start = sentence.find(token, last_index)
            end = start + len(token)
        # print(token,start,end)
        token_positions.append((start, end))
        last_index = end
    print(token_positions)
    # Merge consecutive entities
    for entity in entities:
        
        if entity['entity'].startswith('B-'):
            print(entity['entity'])
            if current_entity:
                print(current_entity, "here")
                merged_entities.append(current_entity)
            
            word = ""
            if entity['word'][:2] == "##":
                word = entity['word'][2:]
            else:
                word = entity['word']
            
            current_entity = {
                'word': word,
                'start': token_positions[entity['index'] - 1][0],
                'end': token_positions[entity['index'] - 1][1],
                'entity': entity['entity'][2:]
            }
        elif entity['entity'].startswith('I-') and current_entity:
            word = ""
            if entity['word'][:2] == "##":
                word = entity['word'][2:]
            else:
                word = entity['word']
            
            current_entity['word'] += word
            current_entity['end'] = token_positions[entity['index'] - 1][1]

    if current_entity:
        merged_entities.append(current_entity)

    # Update positions to match the entire merged word
    for entity in merged_entities:
        entity['start'] = sentence.find(entity['word'])
        entity['end'] = entity['start'] + len(entity['word'])

    return merged_entities

output = merge_entities(op, decoded, exp)
print(output)
