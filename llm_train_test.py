import json
from tqdm import tqdm
import argparse

from datasets import Dataset
from transformers import ( AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM)
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from peft import LoraConfig, PeftModel
from ranx import Qrels, Run, evaluate
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
import string
import pickle as pkl

SEPARATOR = "### Answer:\n"


def preprocess(txt):
    return set(txt.translate(str.maketrans('', '', string.punctuation)).lower().split())

def adjust_length_to_model(tokenizer, max_length, text):
    max_new_tokens = 100
    len_sep = len(tokenizer(SEPARATOR, max_length=max_length, truncation=True))
    tokenized_text = tokenizer(text, max_length=max_length, truncation=True)
    len_tokenized = len(tokenized_text['input_ids'])
    if len_tokenized >= max_length:
        tokens = tokenized_text['input_ids'][:max_length-max_new_tokens-len_sep]
        text = tokenizer.decode(tokens) + SEPARATOR
    return text

def prompt_input_train(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Answer:\n{output}\n").format_map(row)
def prompt_input_dev(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Answer:\n").format_map(row)


def preprocess_data(data_dict, tokenizer, max_length, split='train'):
    # Préparer les textes
    texts = []
    for data in data_dict:
        if split == 'train':
            text = prompt_input_train(data) + tokenizer.eos_token
        else:
            text = prompt_input_dev(data)
        texts.append(text)

    # Créer le Dataset
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(lambda examples: tokenizer(examples["text"], max_length=max_length, truncation=True), batched=True)
    
    # # Créer le ConstantLengthDataset
    # dataset = ConstantLengthDataset(
    #     tokenizer=tokenizer,
    #     dataset=data_dict,
    #     formatting_func=prompt_input,
    #     seq_length=max_length,
    #     infinite=False,
    #     shuffle=True,
    #     append_concat_token=False
    # )
    
    return dataset

def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
     
    return intersection / union

def create_model_config(model_name):
    use_peft = True
    target_modules=["q_proj", "k_proj"]
    modules_to_save=["embed_tokens", "lm_head"]
    model_class = AutoModelForCausalLM
    extra_loading_kwargs = {}

    if model_name == 'llama':
        model_path = 'models/Llama-3.2-1B-Instruct'
        max_length=4096
    elif model_name == 'bloom':
        model_path = '/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-560m'
        target_modules=["query_key_value"]
        max_length=2048
    elif model_name == 'bloomz560':
        model_path = 'models/bloomz-560m'
        target_modules=["query_key_value"]
        max_length=2048
    elif model_name == 'bloomz1b7':
        # Out of memory
        model_path = 'models/bloomz-1b7'
        target_modules=["query_key_value"]
        max_length=2048
    elif model_name == 'bloomz7b1':
        # Out of memory
        model_path = 'models/bloomz-7b1'
        target_modules=["query_key_value"]
        max_length=2048
    elif model_name == 'bloom-gpu4':
        model_path = 'models/bloom-560m'
        target_modules=["query_key_value"]
        max_length=2048
    elif model_name == 'minicpm':
        model_path = 'models/MiniCPM3-4B'
        max_length=2048
        target_modules=["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        extra_loading_kwargs['trust_remote_code'] = True
    elif model_name == 'minicpmRag':
        model_path = 'models/MiniCPM3-RAG-LoRA'
        max_length=2048
        target_modules=["qkv_proj"]
        extra_loading_kwargs['trust_remote_code'] = True
    elif model_name == 'phi3':
        # Out of memory
        model_path = 'models/Phi-3.5-mini-instruct'
        max_length=4096
        target_modules=["qkv_proj"]

    return use_peft, target_modules, modules_to_save, model_path, max_length, model_class, extra_loading_kwargs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--model_name", help="model to use", default="bloom",
                        choices=["bloom", "bloom-gpu4", "bloomz560", "bloomz1b7", "bloomz7b1", "minicpm", "minicpmRag", "phi3", "llama"])
    parser.add_argument("--dataset", help="dataset to test", type=str, default="scifact")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--test_cpt", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print(args)

    model_name = args.model_name
    use_peft, target_modules, modules_to_save, model_path, max_length, model_class, extra_loading_kwargs = create_model_config(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=max_length, padding_side='right', padding='max_length')
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model_class.from_pretrained(model_path, **extra_loading_kwargs)
    print(model)
    
    dataset = args.dataset
    with open(f'datasets/{dataset}_train.json', encoding='utf-8') as f:
        train_instr = json.load(f)
    with open(f'datasets/{dataset}_dev.json', encoding='utf-8') as f:
        raw_dev_instr = json.load(f)

    nb_train_samples = args.steps_per_epoch

    train_instr = preprocess_data(train_instr, tokenizer, max_length, 'train')
    eval_instr = preprocess_data(raw_dev_instr, tokenizer, max_length, 'train')
    dev_instr = preprocess_data(raw_dev_instr, tokenizer, max_length, 'dev')
    data_collator = DataCollatorForCompletionOnlyLM(SEPARATOR, tokenizer=tokenizer)

    
    # 4. Vérification de la configuration
    print(f"Padding side: {tokenizer.padding_side}")
    print(f"Pad token: {tokenizer.pad_token}")

    output_dir=f"checkpoints/{dataset}-cpt-{model_name}-{args.epochs}ep-{args.learning_rate}lr"
    training_args = SFTConfig(
        output_dir=output_dir, 
        overwrite_output_dir=args.overwrite_output_dir, 
        num_train_epochs=args.epochs, 
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=2,  
        eval_steps=nb_train_samples, 
        save_steps=args.save_steps*nb_train_samples,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_seq_length=max_length,
        # load_best_model_at_end=True,
        # save_total_limit=2,
        save_safetensors=False,
        packing=False,
    )

    if use_peft:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",  
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model.add_adapter(peft_config)
        
    if args.train:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_instr,
            eval_dataset=eval_instr,
        )
        trainer.train()
        trainer.save_model()

    
    # test
    gt_classif = []
    pred_classif = []
    pred_classif_inclusif = []
    full_results = []
    jaccard_results = []
    rundict = {}
    corpus_test, queries_test, qrels_test = pkl.load(open(f"dataset_iso.pkl", "rb"))
    for k, v in qrels_test.items():
        qrels_test[k] = {str(list(v.keys())[0]): 1}

    if use_peft:
        model = model_class.from_pretrained(model_path)
        if not args.train and args.test_cpt > 0:
            model = PeftModel.from_pretrained(model=model, model_id=output_dir+f"/checkpoint-{args.test_cpt*nb_train_samples}")
        else:
            model = PeftModel.from_pretrained(model=model, model_id=output_dir)
    model.eval()
    
    test_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda', truncation=True, max_length=max_length, max_new_tokens=1024) 
    
    max_new_tokens = 100
    res_all = tqdm(test_pipeline(KeyDataset(dev_instr, 'text')))
    for res, raw_data in zip(res_all, raw_dev_instr):
        res = res[0]['generated_text']
        response = res.split(SEPARATOR)[1]
        response = response.split(tokenizer.eos_token)[0]
        
        output = preprocess(raw_data['output'])
        gt_classif.append(preprocess(raw_data['output']))
        pred = preprocess(response)
        pred_classif.append(pred)
        jsim = jaccard_similarity(pred, output)
        full_results.append({'jaccard': jsim, 'expected': raw_data['output'], 'full_response': response})
        jaccard_results.append(jsim)

        if response != 'NONE' and dataset == 'iso_none':
            did = [i for i in corpus_test.keys() if corpus_test[i]==raw_data['input']][0]
            jacc_tab = [(k,jaccard_similarity(preprocess(queries_test[k]), pred)) for k in queries_test.keys()]
            jacc_tab = sorted(jacc_tab, key=lambda x: x[1], reverse=True)
            qid = str(jacc_tab[0][0])
            if qid not in rundict:
                rundict[qid] = {}
            rundict[qid][str(did)] = 1


    print("AVG Jaccard Similarity: ", sum(jaccard_results)/len(jaccard_results))
    print("Length of dataset: ", len(jaccard_results))
    print("NB Jaccard Similarity != 0 : ", len([x for x in jaccard_results if x != 0]))
    non_null_jaccard = sorted([x for x in full_results], key=lambda x: x['jaccard'], reverse=True)
    print("Non null Jaccard: \n", non_null_jaccard)

    if dataset == 'iso_none':
        # Evaluate with ranx
        print("\n\n")
        print(qrels_test)
        print(rundict)
        print("\n\n")
        qrels = Qrels(qrels_test)
        run = Run(rundict)
        results = evaluate(qrels, run, ['recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5',
                                        'ndcg@1', 'ndcg@2', 'ndcg@3', 'ndcg@4', 'ndcg@5'], make_comparable=True)
        print(results)
    
    with open(f'json/res_{dataset}-{model_name}-{args.epochs}ep-{args.learning_rate}lr.json', mode='w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
