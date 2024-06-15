import datasets
import random
import time
import pandas as pd
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import dump_jsonl, extract_text_in_double_brackets, get_cohere_gen, write_list_dic_to_csv, extract_number, get_prism_user_map, get_prism_prompt, get_api_response

def get_args_parser():
    parser = argparse.ArgumentParser('prism', add_help=False)
    parser.add_argument("--log_path", default="prism.log", type=str, help="Path to save the log")
    # parser.add_argument("--jsonl_path", default="prism.jsonl", type=str, help="Path to save the jsonl file")
    parser.add_argument("--num_sample", default=1000, type=int)
    parser.add_argument("--prompt_type", choices=["with_persona", "no_persona", "with_persona_with_tie", "no_confidence", "1_to_100"], default="with_persona", type=str)
    parser.add_argument("--persona_features", choices=["all_features", "with_desc", "key_features", "least_imp_feature"], default="all_features", type=str)
    parser.add_argument("--model", choices = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "command-r-plus", "meta-llama/Meta-Llama-3-70B-Instruct"], default="command-r-plus", type=str)
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    jsonl_path = os.path.join("outputs", f"prism_{args.prompt_type}_{args.persona_features}_{args.model}_jun5.jsonl".replace("/", "_"))
    assert not os.path.exists(jsonl_path)
    assert os.path.exists("./outputs")
    random.seed(0)
    if "llama" in args.model:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    else:
        tok = None
        model = args.model
        
    user_data = datasets.load_dataset('HannahRoseKirk/prism-alignment', 'survey', split='train')
    user_map = get_prism_user_map(args.persona_features, user_data)

    chat_data = datasets.load_dataset('HannahRoseKirk/prism-alignment', 'utterances', split='train')
    chat_data = chat_data.filter(lambda example: example['turn'] == 0)
    # chat_data = chat_data.filter(lambda example: example['conversation_type'] == 'values guided')
    if "gpt" in args.model:
        chat_data = chat_data.filter(lambda example: 'gpt' not in example['model_name'])
    elif 'claude' in args.model:
        chat_data = chat_data.filter(lambda example: 'claude' not in example['model_name'])
    elif "command" in args.model:
        chat_data = chat_data.filter(lambda example: 'command' not in example['model_name'])
    chat_data = chat_data.filter(lambda example: 'language model' not in example['model_response'] and 'AI' not in example['model_response'] and "chatbot" not in example['model_response'])
    print(len(chat_data))
    int_ids = list(set(chat_data['interaction_id']))
    int_ids = sorted(int_ids, key=lambda x: int(x[3:]))
    data = []
    count = 0

    for idx, int_id in enumerate(int_ids):
        print("*"*50)
        print(int_id)
        chat_int = chat_data.filter(lambda example: example['interaction_id'] == int_id)
        user_id = chat_int[0]['user_id']
        user_info = user_map[user_id]
        
        tmp = chat_data.filter(lambda example: example['interaction_id']==int_id)
        if len(tmp) < 2:
            print("skipping")
            continue
        sorted_indices = sorted(enumerate(tmp['score']), key=lambda x: x[1], reverse=True)
        max_index = sorted_indices[0][0]
        # second_max_index = sorted_indices[-1][0] ## should be 1 if second max 
        second_max_index = random.choice(sorted_indices[1:])[0]
        print("max score", tmp['score'][max_index], "second max score", tmp['score'][second_max_index])
        ## assert the max score > second max score + 10
        question = tmp[0]['user_prompt']
        chosen = tmp['model_response'][max_index]
        rejected = tmp['model_response'][second_max_index]
        
        if "tie" in args.prompt_type:
            if tmp['score'][max_index] < tmp['score'][second_max_index] + 10:
                gt = "C"
            else:
                random_number = random.randint(0, 1)
                if random_number > 0.5:
                    asst_A = chosen
                    asst_B = rejected
                    gt = "A"
                else:
                    asst_A = rejected
                    asst_B = chosen
                    gt = "B"
        
        else:
            if tmp['score'][max_index] < tmp['score'][second_max_index] + 10:
                print("skipping")
                continue

            random_number = random.randint(0, 1)
            if random_number > 0.5:
                asst_A = chosen
                asst_B = rejected
                gt = "A"
            else:
                asst_A = rejected
                asst_B = chosen
                gt = "B"
            
        prompt = get_prism_prompt(args.prompt_type, user_info, question, asst_A, asst_B)
        print(prompt)
        res = get_api_response(prompt, model=model, tokenizer=tok, max_tokens=15)
        ans = res.replace("[[", "").replace("Answer:", "").strip()[0]
        acc = (ans == gt)
        print(ans, gt)
        # import pdb; pdb.set_trace()
        try:
            certainty = extract_number(res)
            dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": user_info, "question": question, "asst_A": asst_A, "asst_B": asst_B, "conversation_type": tmp['conversation_type'][0]}
            data.append(dict)
            dump_jsonl(dict, jsonl_path)
        except:
            if "tie" in args.prompt_type or "no_persona" in args.prompt_type:
                certainty = "NA"
                dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": user_info, "question": question, "asst_A": asst_A, "asst_B": asst_B, "conversation_type": tmp['conversation_type'][0]}
                dump_jsonl(dict, jsonl_path)
            else:
                print("Error")
        
        count += 1
        if count > args.num_sample:
            break
        

    df = pd.read_json(jsonl_path, lines=True)
    df['certainty'] = df['certainty'].apply(lambda x: int(x//10))
    df = df[(df['answer'] == "A") | (df['answer'] == "B")]
    
    # if df['certainty'][0] > 50:
    #     df['certainty'] = df['certainty'].apply(lambda x: int(x//10))

    grouped = df.groupby('certainty')['acc'].agg(Total_Responses='count', Correct_Responses=lambda x: x.sum(), Accuracy='mean')
    print(grouped)
    with open(args.log_path, 'a') as f:
        f.write(f"Model: {args.model}, Prompt Type: {args.prompt_type}, Persona Features: {args.persona_features}, Num Sample: {args.num_sample}\n")
        f.write(f"{grouped}\n")
        ## Total Acc
        f.write(f"\nTotal Accuracy: {df['acc'].mean()}\n")
        f.write("\n-------------------------------------\n")
    print("done")