import json
import numpy as np
import random
import os
import torch
import pandas as pd
import argparse
from utils import dump_jsonl, extract_number, get_api_response
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


data = []
encodings = []
count = 0

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_args_parser():
    parser = argparse.ArgumentParser('opinionqa', add_help=False)
    parser.add_argument("--log_path", default="ap.log", type=str, help="Path to save the log")
    parser.add_argument("--num_sample", default=600, type=int)
    parser.add_argument("--prompt_type", choices=["with_persona", "no_persona", "no_confidence"], default="with_persona", type=str)
    parser.add_argument("--persona_features", choices=["all_features", "least_imp_feature", "key_features"], default="all_features", type=str)
    parser.add_argument("--model", choices = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "command-r-plus", "meta-llama/Meta-Llama-3-70B-Instruct"], default="command-r-plus", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    return parser
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    jsonl_path = os.path.join(args.output_dir, f"ap_{args.prompt_type}_{args.persona_features}_{args.model}.jsonl".replace("/", "_"))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data_path = "./ap/synthetic_dataset.jsonl"
    print("Loading model")
    # sent_bert_tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    sent_bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
    print("Model loaded")
    
    assert not os.path.exists(jsonl_path)
    if "llama" in args.model:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    else:
        tok = None
        model = args.model
    
    with open(data_path, 'r') as f:
        tmp = f.readlines()
        for idx, dp in enumerate(tmp):
            persona = []
            dp = json.loads(dp)  
            persona.append(f"The person is {dp['personality']['age']} years old")
            persona.append(f"The person is {dp['personality']['sex']}")
            persona.append(f"The person is living at {dp['personality']['city_country']}")
            persona.append(f"The person is born in {dp['personality']['birth_city_country']}")
            persona.append(f"The person's highest education level is {dp['personality']['education']}")
            persona.append(f"The person's occupation is {dp['personality']['occupation']}")
            persona.append(f"The person's income is {dp['personality']['income']}")
            persona.append(f"The person is {dp['personality']['relationship_status']}")
            persona = ". ".join(persona) + "."
            dp['desc'] = persona
            # import pdb; pdb.set_trace()
            encoding = sent_bert_model.encode([persona])[0]
            data.append(dp)
            print(idx)
            
    for dp in data:
        similar_persona_idx = ""
        similarity = 0
        ### Find the most similar persona
        for idx, key in enumerate(encodings):
            sim = cosine_similarity(dp['encoding'], key)
            if sim > similarity:
                similarity = sim
                similar_persona_idx = idx
        similar_persona = data[idx]['desc']
        similar_response = data[idx]['response']
        similar_question = data[idx]['question_asked']
        
        if random.random() > 0.5:
            response_A = dp['response']
            response_B = similar_response
            question_A = dp['question_asked']
            question_B = similar_question
            gt = "A"
        else:
            response_A = similar_response
            response_B = dp['response']
            question_A = similar_question
            question_B = dp['question_asked']
            gt = "B"
        
        prompt = """Given the user profile provided below, select the question-answer pair of which the answer is most likely written by the user. Declare your choice by using the format: "[[A]]" if you believe Answer A is more suitable, or "[[B]]" if Answer B is better suited. Additionally, assess your confidence in this decision by assigning a certainty level from 1 to 100. Use the following guidelines to assign the certainty level:

1--20 (Uncertain): The user profile provides insufficient or Minimal evidence information suggests a preference. The decision is largely based on weak or indirect hints.
21--40 (Moderately Confident): There is noticeable evidence supporting a preference, though it is not comprehensive, and other interpretations are possible.
41--60 (Quite Confident): You find clear and convincing evidence that supports your prediction, though it is not entirely decisive.
61--80 (Confident): The user profile contains strong evidence that clearly supports your prediction, with very little ambiguity.
81--100 (Highly Confident): The user profile provides direct and explicit evidence that decisively supports your prediction.
Ensure you enclose your chosen certainty level in double brackets, like so: [[X]].

[User Profile]
{user_info}

[Question-Answer Pair A]
Question: {question_A}
Answer: {response_A}

[Question-Answer Pair B]
Question: {question_B}
Answer: {response_B}

[Answer]
[[""".format(user_info=dp['desc'], question_A=question_A, response_A=response_A, question_B=question_B, response_B=response_B)

#         prompt = """For the questiona and response below, select the user profile that most likely respond to the question in that way. Declare your choice by using the format: "[[A]]" if you believe User A is more suitable, or "[[B]]" if User B is better suited. Additionally, assess your confidence in this decision by assigning a certainty level from 1 to 100. Use the following guidelines to assign the certainty level:

# 1--20 (Uncertain): The user profile provides insufficient or Minimal evidence information suggests a preference. The decision is largely based on weak or indirect hints.
# 21--40 (Moderately Confident): There is noticeable evidence supporting a preference, though it is not comprehensive, and other interpretations are possible.
# 41--60 (Quite Confident): You find clear and convincing evidence that supports your prediction, though it is not entirely decisive.
# 61--80 (Confident): The user profile contains strong evidence that clearly supports your prediction, with very little ambiguity.
# 81--100 (Highly Confident): The user profile provides direct and explicit evidence that decisively supports your prediction.
# Ensure you enclose your chosen certainty level in double brackets, like so: [[X]].
    
# [Question]
# {question}

# [User Response]
# {response}

# [User A's Profile]
# {user_info_a}

# [User B's Profile]
# {user_info_b}

# [Answer]
# [[""".format(question=dp['question_asked'], response=dp['response'], user_info_a=similar_persona, user_info_b=dp['desc'])

        # res = get_api_response(prompt, model=model, tokenizer=tok, max_tokens=15)
        # import pdb; pdb.set_trace()
        # ans = res.replace("(", "").replace("[[", "").replace("Answer:", "").strip()[0]
        
        def parse_res(res):
            ans = res.replace("(", "").replace("[[", "").replace("Answer:", "").strip()[0]
            if ans != "A" and ans != "B":
                return False, False
            try:
                certainty = extract_number(res)
                return ans, certainty
            except:
                return False, False
        
        def get_api_response_with_parse(prompt, model, tok=None, max_tokens=15, temperature = 0.7, stop_strs = None, max_depth = 3, cur_depth = 0):
            res = get_api_response(prompt, model=model, tokenizer=tok, max_tokens=max_tokens)
            ans, certainty = parse_res(res)
            if ans == False and cur_depth < max_depth:
                print("regenerating")
                return get_api_response_with_parse(prompt, model=model, tok=tok, max_tokens=max_tokens, temperature = temperature, stop_strs = stop_strs, max_depth = max_depth, cur_depth = cur_depth+1)
            if cur_depth > 0 and cur_depth < max_depth:
                print("regeneration succeed")
            elif cur_depth == max_depth:
                print("regeneration failed")
            return ans, certainty, res
        
        ans, certainty, res = get_api_response_with_parse(prompt, model=model, tok=tok, max_tokens=15)
        acc = (ans == gt)
        print("res: ", res)
        print("ans: ", ans, gt)
        dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "res": res}
        dump_jsonl(dict, jsonl_path)
        
        # try:
        #     certainty = extract_number(res)
        #     print("certainty: ", certainty)
        #     dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "res": res, "similar_persona": similar_persona, "similar_response": similar_response, "similar_question": similar_question}
        #     dump_jsonl(dict, jsonl_path)
        # except:
        #     if "tie" in args.prompt_type or "no_persona" in args.prompt_type:
        #         certainty = "NA"
        #         dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "res": res}
        #         dump_jsonl(dict, jsonl_path)
        #     else:
        #         print("Error")
        count += 1
        if count > args.num_sample:
            break
        
    df = pd.read_json(jsonl_path, lines=True)
    df = df[(df['answer'] == "A") | (df['answer'] == "B")]
    if df['certainty'][0] > 50:
        df['certainty'] = df['certainty'].apply(lambda x: int(x//10))
        
    grouped = df.groupby('certainty')['acc'].agg(Total_Responses='count', Correct_Responses=lambda x: x.sum(), Accuracy='mean')
    with open(args.log_path, 'a') as f:
        f.write(f"Model: {args.model}, Prompt Type: {args.prompt_type}, Persona Features: {args.persona_features}, Num Sample: {args.num_sample}\n")
        f.write(f"{grouped}\n")
        ## Total Acc
        f.write(f"\nTotal Accuracy: {df['acc'].mean()}\n")
        f.write("\n-------------------------------------\n")
    print("done")