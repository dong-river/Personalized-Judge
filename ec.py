import os
import pandas as pd
import itertools
import ast
import argparse
import random
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import dump_jsonl, extract_text_in_double_brackets, get_cohere_gen, write_list_dic_to_csv, extract_number, get_api_response

def get_prompt(prompt_type, user_info, article, asst_A, asst_B):
    if prompt_type == "no_confidence":
        prompt = """Based on the user profile and the news article provided below, identify which response—A or B—most closely aligns with the style and preferences indicated by the user profile. Declare your choice by using the format: "[[A]]" if you believe Response A is more suitable, or "[[B]]" if Response B better suited. 

[News Article]
{article}

[User Profile]
{user_info}

[Response A]
{asst_A}

[Response B]
{asst_B}

[Answer]
[[""".format(user_info=user_info, article=article, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "no_persona":
        prompt = """Given the news article provided below, select the response that is the most reasonable response to the news. Declare your choice by using the format: "[[A]]" if you believe Response A is more suitable, or "[[B]]" if Response B better suited. 

[News Article]
{article}

[Response A]
{asst_A}

[Response B]
{asst_B}

[Answer]
[[""".format(article=article, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "with_persona":
        prompt = """Based on the user profile provided, we have two responses from the user to a news article. Your task is to determine which response aligns better with the user's profile.
[News Article]
{article}
[User Profile]
{user_info}
[Response A]
{asst_A}
[Response B]
{asst_B}
Evaluate the responses and select the one you believe is written by the user. Use the format "[[A]]" if you think Response A is more appropriate, or "[[B]]" if Response B is a better match. Additionally, assess your confidence in this decision by assigning a certainty level from 1 to 100 according to the scale below. Enclose your chosen certainty level in double brackets, like so: [[X]].

Certainty Scale:
1--20 (Uncertain): The user profile provides insufficient or Minimal evidence information suggests a preference. The decision is largely based on weak or indirect hints.
21--40 (Moderately Confident): There is noticeable evidence supporting a preference, though it is not comprehensive, and other interpretations are possible.
41--60 (Quite Confident): You find clear and convincing evidence that supports your prediction, though it is not entirely decisive.
61--80 (Confident): The user profile contains strong evidence that clearly supports your prediction, with very little ambiguity.
81--100 (Highly Confident): The user profile provides direct and explicit evidence that decisively supports your prediction.

[Answer]
[[""".format(user_info=user_info, article=article, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "with_persona_with_tie":
        prompt = """Given the user profile provided below, select the response that the user would most likely prefer. Declare your choice by using the format: "[[A]]" if you believe assistant A's response is more suitable, "[[B]]" if assistant B's response is better suited, or "[[C]]" for a tie.
[News Article]
{article}
[User Profile]
{user_info}
[Response A]
{asst_A}
[Response B]
{asst_B}
[Answer]
[[""".format(user_info=user_info, article=article, asst_A=asst_A, asst_B=asst_B)
    else:
        raise NotImplementedError
    return prompt
    

def get_args_parser():
    parser = argparse.ArgumentParser('ec', add_help=False)
    parser.add_argument("--log_path", default="ec.log", type=str, help="Path to save the log")
    parser.add_argument("--num_article", default=100, type=int)
    parser.add_argument("--num_pair_per_article", default=50, type=int)
    parser.add_argument("--prompt_type", choices=["with_persona", "with_persona_with_tie", "no_persona", "no_confidence"], default="with_persona", type=str)
    parser.add_argument("--persona_features", choices=["all_features"], default="all_features", type=str)
    parser.add_argument("--model", choices = ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "gemini-1.5-pro-latest", "command-r-plus", "meta-llama/Meta-Llama-3-70B-Instruct"], default="command-r-plus", type=str)
    return parser
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    jsonl_path = f"outputs/ec_{args.prompt_type}_{args.model}.jsonl".replace("/", "_")
    jsonl_path = os.path.join("outputs", f"ec_{args.prompt_type}_{args.persona_features}_{args.model}.jsonl".replace("/", "_"))
    if "llama" in args.model:
        tok = AutoTokenizer.from_pretrained(args.model, token="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")
        model = AutoModelForCausalLM.from_pretrained(args.model, token="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", device_map="auto", torch_dtype=torch.float16)
    else:
        tok = None
        model = args.model

    annotator_df = pd.read_csv("ec/trac4_PER_train.csv")
    df = pd.read_csv("ec/trac3_EMP_train.csv", on_bad_lines='skip')
    article_df = pd.read_csv("ec/articles_adobe_AMT.csv")

    gender_map = {1: 'Male', 2: 'Female', 5: 'Other'}
    race_map = {1: 'White', 2: 'Hispanic / Latino', 3: 'Black / African American', 4: 'Native American / American Indian', 5: 'Asian / Pacific Islander', 6: 'Other'}
    education_map = {1: 'a diploma less than a high school', 2: 'High school degree or diploma', 3: 'went to Technical / Vocational School', 4: 'went to college but did not get a degree', 5: 'Two year associate degree', 6: 'College or university degree', 7: 'Postgraduate / professional degree'}

    annotator_dict = {}
    tie_count = 0
    for idx, row in annotator_df.iterrows():
        persona = []
        persona.append(f"The person is {gender_map[row['gender']].lower()}.")
        persona.append(f"Racially, the person is {race_map[row['race']].lower()}.")
        persona.append(f"The person is {row['age']} years old.")
        persona.append(f"The person has a {education_map[row['education']].lower()}.")
        persona.append(f"The person earns {row['income']} dollar per year.")
        persona.append(f"According to the Big Five personality test, on a scale of 10, the person has scored {row['personality_openess']} in openness, {row['personality_conscientiousness']} in conscientiousness, {row['personality_extraversion']} in extraversion, {row['personality_agreeableness']} in agreeableness, and {row['personality_stability']} in stability.")
        persona = [p for p in persona if "nan" not in p]
        persona = " ".join(persona)
        annotator_dict[row['person_id']] = persona

    article_ids = list(set(df['article_id']))[:args.num_article]
    for article_id in article_ids:
        count = 0
        tmp = df[df['article_id'] == article_id]
        article = article_df[article_df['article_id'] == article_id]['text'].values[0]

        
        pairs = list(itertools.combinations(tmp.iterrows(), 2))
        for (index1, row1), (index2, row2) in pairs:
            if count > args.num_pair_per_article:
                print("done with this article")
                break
            if "tie" not in args.prompt_type:
                if abs(row1['person_empathy'] - row2['person_empathy']) < 3 or abs(row1['person_distress'] - row2['person_distress']) < 3:
                    print("skipping")
                    continue
                random_number = random.randint(0, 1)
                if random_number > 0.5:
                    asst_A = row1['person_essay']
                    asst_B = row2['person_essay']
                    gt = "A"
                else:
                    asst_A = row2['person_essay']
                    asst_B = row1['person_essay']
                    gt = "B"
            else:
                random_number = random.randint(0, 1)
                if random_number > 0.5:
                    asst_A = row1['person_essay']
                    asst_B = row2['person_essay']
                    gt = "A"
                else:
                    asst_A = row2['person_essay']
                    asst_B = row1['person_essay']
                    gt = "B"
                if abs(row1['person_empathy'] - row2['person_empathy']) < 3 or abs(row1['person_distress'] - row2['person_distress']) < 3:
                    gt = "C"
                    if tie_count > 100:
                        continue
                    tie_count += 1
                    
                
            count += 1
            persona = annotator_dict[row1['person_id']]
            prompt = get_prompt(args.prompt_type, user_info=persona, article=article, asst_A=asst_A, asst_B=asst_B)
            print(prompt)
            res = get_api_response(prompt, model=model, tokenizer=tok, max_tokens=15)
            ans = res.replace("(", "").replace("[[", "").replace("Answer:", "").strip()[0]
            acc = (ans == gt)
            print(acc, ans, gt)
            try:
                certainty = extract_number(res)
                print(certainty)
                dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "article": article, "asst_A": asst_A, "asst_B": asst_B}
                dump_jsonl(dict, jsonl_path)
            except:
                if "tie" in args.prompt_type or "no_persona" in args.prompt_type:
                    certainty = "NA"
                    dict = {"prompt": prompt, "answer": ans, "certainty": certainty, 'acc': acc, "ground_truth": gt, "user_info": persona, "article": article, "asst_A": asst_A, "asst_B": asst_B}
                    dump_jsonl(dict, jsonl_path)
                else:
                    print("Error")
        
    df = pd.read_json(jsonl_path, lines=True)
    df = df[(df['answer'] == "A") | (df['answer'] == "B")]
    if df['certainty'][0] > 50:
        df['certainty'] = df['certainty'].apply(lambda x: int(x//10))
    grouped = df.groupby('certainty')['acc'].agg(Total_Responses='count', Correct_Responses=lambda x: x.sum(), Accuracy='mean')
    with open(args.log_path, 'a') as f:
        f.write(f"Model: {args. model}, Prompt Type: {args.prompt_type}, Persona Features: {args.persona_features})\n")
        f.write(f"{grouped}\n")
        ## Total Acc
        f.write(f"\nTotal Accuracy: {df['acc'].mean()}\n")
        f.write("\n-------------------------------------\n")
    print("done")