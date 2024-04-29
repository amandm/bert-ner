from transformers import pipeline
import gradio as gr

from transformers import BertJapaneseTokenizer, BertForTokenClassification

MODEL_DIR = "./model_v2/checkpoint-18000/"
model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR,model_max_length=128)
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

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

examples = [
    "株式会社はJurabi、東京都台東区に本社を置くIT企業である","レッドフォックス株式会社は、東京都千代田区に本社を置くITサービス企業である"
]


def ner(text):
    entities = ner_pipeline(text)
    # print(entities)
    for i in entities:
        print(i)
    encoding = tokenizer(text)
    # print(tokens)
    decoded = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    print(decoded)
    
    output = merge_entities(entities, decoded, text)
    # text = "レッドフォックス株式会社は、東京都千代田区に本社を置くITサービス企業である。"
    # output = [ { "word": "レッドフォックス株式会社", "start": 0,"end": 12, "entity": "法人名" }, { "word": "東京都千代田区", "start": 14,"end": 21, "entity": "地名" } ]
    # output = [{'word': 'レッドフォックス株式会社', 'start': 1, 'end': 5, 'entity': '法人名'}, {'word': '東京都千代田区', 'start': 7, 'end': 10, 'entity': '地名'}]
    return {"text": text, "entities": output}    

demo = gr.Interface(ner,
             gr.Textbox(placeholder="Enter sentence here..."), 
             gr.HighlightedText(),
             examples=examples)

demo.launch()