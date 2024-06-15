import os
import argparse
import pandas as pd
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import dump_jsonl, get_api_response, extract_number

def create_persona(row, persona_features="all_features"):
    if persona_features == "all_features":
        persona = []
        persona.append(f"Racially, the person is {str(row['RACE']).lower()}.")
        persona.append(f"The person lives in the {str(row['CREGION']).lower()} region.")
        persona.append(f"The person is in the {str(row['AGE'])} age group.")
        persona.append(f"The person is {str(row['SEX']).lower()}.")
        persona.append(f"The person's highest level of education is {str(row['EDUCATION']).lower()}.")
        persona.append(f"The person is {'a citizen' if str(row['CITIZEN']) == 'Yes' else 'not a citizen'} of the United States.")
        persona.append(f"The person is {str(row['MARITAL']).lower()}.")
        persona.append(f"The person follows the {str(row['RELIG']).lower()} religion and attends religious services {str(row['RELIGATTEND']).lower()}.")
        persona.append(f"Politically, the person aligns with the {str(row['POLPARTY']).lower()} party and considers themselves {str(row['POLIDEOLOGY']).lower()}.")
        persona.append(f"The person earns {str(row['INCOME']).lower()} per year.")
        persona = [p for p in persona if "nan" not in p]
        return " ".join(persona)
    elif persona_features == "least_imp_feature":
        persona = []
        persona.append(f"The person is {str(row['MARITAL']).lower()}.")
        persona = [p for p in persona if "nan" not in persona]
        return " ".join(persona)
    elif persona_features == "key_features":
        persona = []
        persona.append(f"The person is in the {str(row['AGE'])} age group.")
        persona.append(f"The person's highest level of education is {str(row['EDUCATION']).lower()}.")
        persona.append(f"Politically, the person aligns with the {str(row['POLPARTY']).lower()} party and considers themselves {str(row['POLIDEOLOGY']).lower()}.")
        # persona.append(f"Racially, the person is {str(row['RACE']).lower()}.")
        # persona.append(f"The person lives in the {str(row['CREGION']).lower()} region.")
        # persona.append(f"The person's highest level of education is {str(row['EDUCATION']).lower()}.")
        persona = [p for p in persona if "nan" not in persona]
        return " ".join(persona)

def get_args_parser():
    parser = argparse.ArgumentParser('opinionqa', add_help=False)
    parser.add_argument("--log_path", default="opinionqa.log", type=str, help="Path to save the log")
    parser.add_argument("--num_sample", default=200, type=int)
    parser.add_argument("--prompt_type", choices=["with_persona", "no_persona", "no_confidence"], default="with_persona", type=str)
    parser.add_argument("--persona_features", choices=["all_features", "least_imp_feature", "key_features"], default="all_features", type=str)
    parser.add_argument("--model", choices = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "command-r-plus", "meta-llama/Meta-Llama-3-70B-Instruct"], default="command-r-plus", type=str)
    parser.add_argument("--cont", action='store_true')
    parser.add_argument("--output_dir", default="outputs", type=str)
    return parser
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    data_dir = "opinions_qa/data/human_resp"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    jsonl_path = os.path.join(args.output_dir, f"opinionqa_{args.prompt_type}_{args.persona_features}_{args.model}_jun_11.jsonl".replace("/", "_"))
    if not args.cont:
        assert not os.path.exists(jsonl_path)
    
    if args.cont:
        df = pd.read_json(jsonl_path, lines=True)
        df['question'] = df['prompt'].apply(lambda x: x.split("[Question]")[1].split("(A)")[0].strip())
        q_set = set(df['question'])
        
    if "llama" in args.model:
        tok = AutoTokenizer.from_pretrained(args.model, token="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")
        model = AutoModelForCausalLM.from_pretrained(args.model, token="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", device_map="auto", torch_dtype=torch.float16)
    else:
        tok = None
        model = args.model
    
    for folder_dir in os.listdir(os.path.join(data_dir)):
            print(folder_dir)
            count = 0
            if "disagreement" in folder_dir or "npy" in folder_dir:
                continue
            info_path = os.path.join(data_dir, folder_dir, "info.csv")
            res_path = os.path.join(data_dir, folder_dir, "responses.csv")

            info_df = pd.read_csv(info_path)
            info_df = info_df[info_df['option_mapping'].apply(lambda x: len(ast.literal_eval(x)) <= 3)].reset_index(drop=True)
            print("len(info_df)",len(info_df))
            res_df = pd.read_csv(res_path)
            res_df['persona'] = res_df.apply(create_persona, persona_features=args.persona_features, axis=1)
            # res_df = res_df[:args.num_sample]

            if len(info_df) == 0:
                continue
            key = info_df['key'][0]
            q = info_df['question'][0]
            if args.cont:
                if q in q_set:
                    print("Skip b/c already done")
                    continue
            print(key, q)
            option_mapping = ast.literal_eval(info_df['option_mapping'][0])

            for index, row in res_df.iterrows():
                gt = row[key]
                if gt == option_mapping[1.0]:
                    gt = "A"
                elif gt == option_mapping[2.0]:
                    gt = "B"
                else:
                    print("Skip b/c not Yes or No")
                    continue
                persona = row['persona']
                if persona == "Error":
                    print("Skip b/c Error")
                    continue
                if args.prompt_type == "with_persona":
                    prompt = """Given the user profile provided below, infer how the user will answer the following question by choosing A or B. Write the answer in parathesis. Additionally, assess your confidence in this decision by assigning a certainty level from 1 to 100. Use the following guidelines to assign the certainty level:

1--20 (Uncertain): The user profile provides insufficient or minimal evidence information suggests a preference. The decision is largely based on weak or indirect hints.
21--40 (Moderately Confident): There is noticeable evidence supporting a preference, though it is not comprehensive, and other interpretations are possible.
41--60 (Quite Confident): You find clear and convincing evidence that supports your prediction, though it is not entirely decisive.
61--80 (Confident): The user profile contains strong evidence that clearly supports your prediction, with very little ambiguity.
81--100 (Highly Confident): The user profile provides direct and explicit evidence that decisively supports your prediction.
Please indicate your answer by writing (A) or (B). Then, you enclose your chosen certainty level in double brackets, like so: [[X]].

[User Profile]
{persona}

[Question]
{question}
(A) {ans_A}
(B) {ans_B}

[Answer]
(""".format(persona=persona, question=q, ans_A=option_mapping[1.0], ans_B=option_mapping[2.0])
                elif args.prompt_type == "no_persona":
                    prompt = """Answer the question below.  Write the answer in parathesis. 
[Question]{question}
(A) {ans_A}
(B) {ans_B}

[Answer]
(""".format(question=q, ans_A=option_mapping[1.0], ans_B=option_mapping[2.0])
                elif args.prompt_type == "no_confidence":
                    prompt = """Given the user profile provided below, directly infer how the user will answer the following question by choosing A or B.  Write the answer in parathesis. 

[User Profile]
{persona}

[Question]
{question}
(A) {ans_A}
(B) {ans_B}

[Answer]
(""".format(persona=persona, question=q, ans_A=option_mapping[1.0], ans_B=option_mapping[2.0])
                # res = get_api_response(prompt, model=model, tokenizer=tok, max_tokens=15)
                
                
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
                
                ans_A = option_mapping[1.0]
                ans_B = option_mapping[2.0]
                # if ans_A.lower() in res.lower():
                #     ans = "A"
                # elif ans_B.lower() in res.lower():
                #     ans = "B"
                    
                acc = (ans == gt)
                print("res: ", res)
                print("ans: ", ans, gt)
                dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "res": res}
                dump_jsonl(dict, jsonl_path)
                
                # try:
                #     certainty = extract_number(res)
                #     print("certainty: ", certainty)
                #     dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "res": res}
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
