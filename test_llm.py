from llamafactory.chat import ChatModel
from src.llamafactory.extras.misc import torch_gc

import json
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
import string

config = {
            "bloom" : { "path": "/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-560m",
                        "adapter_path": "lora/bloom_lora_sf_10",
                        "template": "default"
                },
            "llama" : { "path": "llama-3-8b-Instruct-bnb-4bit",
                        "adapter_path": "lora/llama_lora",
                        "template": "llama3"
                },
            "phi" : { "path": "phi-1_5",
                        "adapter_path": "lora/phi_lora",
                        "template": "phi"
                },
            "mistral" : { "path": "Mistral-7B-Insutrct-v0.2",
                        "adapter_path": "lora/mistral_lora",
                        "template": "mistral"
                }
    }

def preprocess(txt):
    return set(txt.translate(str.maketrans('', '', string.punctuation)).lower().split())

def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
     
    return intersection / union

if __name__ == '__main__':
    torch_gc()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model to use", default="bloom")
    parser.add_argument("--dataset", help="dataset to test", default="scifact")
    args = parser.parse_args()
    model_name = args.model_name
    args_llm = dict(
      model_name_or_path=config[model_name]["path"], # use bnb-4bit-quantized Llama-3-8B-Instruct model
      adapter_name_or_path=config[model_name]["adapter_path"],            # load the saved LoRA adapters
      template=config[model_name]["template"],                     # same to the one in training
      finetuning_type="lora",                  # same to the one in training
      quantization_bit=4,                    # load 4-bit quantized model
    )
    chat_model = ChatModel(args_llm)

    gt_classif = []
    pred_classif = []
    pred_classif_inclusif = []
    full_results = []
    jaccard_results = []
    
    dataset = args.dataset
    with open(f'llm_dataset/{dataset}_dev.json', encoding='utf-8') as f:
        dev_instr = json.load(f)

    for instr in tqdm(dev_instr):
        messages = []
        query = instr['instruction'] + "\n" + instr['input']

        messages.append({"role": "user", "content": query})
        output = preprocess(instr['output'])
        gt_classif.append(preprocess(instr['output']))

        response = ""
        for new_text in chat_model.stream_chat(messages):
            response += new_text
        pred = preprocess(response)
        pred_classif.append(pred)
        jsim = jaccard_similarity(pred, output)
        full_results.append({'jaccard': jsim, 'expected': instr['output'], 'full_response': response})
        jaccard_results.append(jsim)

        torch_gc()

    print("AVG Jaccard Similarity: ", sum(jaccard_results)/len(jaccard_results))
    print("Length of dataset: ", len(jaccard_results))
    print("NB Jaccard Similarity != 0 : ", len([x for x in jaccard_results if x != 0]))
    non_null_jaccard = sorted([x for x in full_results], key=lambda x: x['jaccard'], reverse=True)
    print("Non null Jaccard: \n", non_null_jaccard)


    with open(f'res_{model_name}_{dataset}_dev.json', mode='w', encoding='utf-8') as f:
        json.dump(full_results, indent=2, ensure_ascii=False)
                
                
