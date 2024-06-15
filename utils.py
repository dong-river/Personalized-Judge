import openai
from openai import OpenAI
import time
import random
import math
import csv
import json
import re
import pandas as pd
from anthropic import Anthropic
import cohere
openai_api_key = "Put your API key here"
client = OpenAI(api_key=openai_api_key)

cohere_api_key = "Put your API key here"
co = cohere.Client(cohere_api_key)

def dump_jsonl(dic, path):
    with open(path, 'a') as outfile:
        json.dump(dic, outfile)
        outfile.write('\n')
        
def write_list_dic_to_csv(data, filename):
    # Writing to csv file
    with open(filename, 'w', newline='') as csvfile:
        # Specifying the fieldnames (keys of the dictionary)
        fieldnames = data[0].keys()
        
        # Creating a DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Writing headers (column names)
        writer.writeheader()
        
        # Writing data
        for row in data:
            writer.writerow(row)
        
def extract_text_in_double_brackets(text):
    try:
        pattern = r'\[\[([^\]]+)\]\]'
        # Find all matches and return them as a list
        return re.findall(pattern, text)
    except Exception as e:
        return f"An error occurred: {e}"
    
def extract_text_in_parenthesis(text):
    try:
        pattern = r'\(([^)]+)\)'
        # Find all matches and return them as a list
        return re.findall(pattern, text)
    except Exception as e:
        return f"An error occurred: {e}"

def get_api_response(prompt, model, tokenizer = None, max_tokens = 10, temperature = 0.7, stop_strs = None, max_depth = 3, cur_depth = 0):
    if type(model) != str:
        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        output_ids = model.generate(tokenized_chat, max_new_tokens=max_tokens, temperature=temperature)
        output = tokenizer.decode(output_ids[0, tokenized_chat.shape[1]:])
        # output = output.replace(prompt, "")
        return output
    elif "command" in model:
        return get_cohere_gen(prompt, model = model, max_tokens = max_tokens, temperature = temperature, stop_strs = stop_strs, max_depth = max_depth, cur_depth = cur_depth)
    elif "gpt" in model:
        return get_openai_gen(prompt, model = model, max_tokens = max_tokens, temperature = temperature, stop_strs = stop_strs, max_depth = max_depth, cur_depth = cur_depth)
    elif "claude" in model:
        return get_claude_gen(prompt, model = model, max_tokens = max_tokens, temperature = temperature, stop_strs = stop_strs, max_depth = max_depth, cur_depth = cur_depth)
    elif "gemini" in model:
        return "API not supported in UK. Need VPN" 
    else:
        raise NotImplementedError

def get_cohere_gen(prompt, system_prompt = "", model = 'command-r-plus', max_tokens = 2048, temperature = 0.7, stop_strs = None, max_depth = 3, cur_depth = 0):
    try:
        if type(prompt) == list:
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} for idx, p in enumerate(prompt)]
            raise NotImplementedError
        elif type(prompt) == str:
            response = co.chat(
                model=model,
                message=prompt
            )
        return response.text
    except Exception as e:
        print(e)
        time.sleep(30)
        return get_cohere_gen(prompt, cur_depth=cur_depth + 1)

def get_claude_gen(prompt, system_prompt = "", model = 'claude-3-sonnet-20240229', max_tokens = 256, temperature = 1, stop_strs = None, max_depth = 3, cur_depth = 0):
    try:
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."
        if type(prompt) == list:
            ## In this case, make sure the prompt list is in the correct order: user, assistant, user, assistant, ...
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} for idx, p in enumerate(prompt)]
        elif type(prompt) == str:
            messages = [{"role": "user", "content": prompt}]
        
        response = claude_client.messages.create(
            model=model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = response.content[0].text
        text = re.sub(r'^Here[^:]*:\s*', '', text.strip())
        text = text.strip()
        return text
    except Exception as e:
        print("Rate Limit Reached. Waiting for 30 seconds")
        print(e)
        time.sleep(60)
        return get_claude_gen(prompt, cur_depth=cur_depth + 1)

def get_openai_logit(prompt, system_prompt = "", model = 'gpt-3.5-turbo', max_tokens = 1, temperature = 0.7, stop_strs = None, max_depth = 3, cur_depth = 0):
    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop_strs,
                temperature=temperature,
                logprobs=True,
                top_logprobs=5
            )
        tokens = [response.choices[0].logprobs.content[0].top_logprobs[i].token for i in range(4)]
        probs = [math.exp(response.choices[0].logprobs.content[0].top_logprobs[i].logprob) for i in range(4)]
    except Exception as e:
        print(e)
        time.sleep(30)
        return get_openai_logit(prompt, cur_depth=cur_depth + 1)
    
    return tokens, probs

def get_openai_gen(prompt, system_prompt = "", model = 'gpt-3.5-turbo', max_tokens = 2048, temperature = 0.7, stop_strs = None, max_depth = 3, cur_depth = 0):
    try:
        if cur_depth >= max_depth:
            return "Sorry, I am not able to answer that question."
        if type(prompt) == list:
            ## In this case, make sure the prompt list is in the correct order: user, assistant, user, assistant, ...
            messages = [{"role": "user", "content": p} if idx % 2 == 0 else {"role": "assistant", "content": p} for idx, p in enumerate(prompt)]
        elif type(prompt) == str:
            messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_strs,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        time.sleep(30)
        return get_openai_gen(prompt, cur_depth=cur_depth + 1)

def extract_number(text):
    number = re.findall(r'\d+', text)[0]
    return number

def get_prism_prompt(prompt_type, user_info, question, asst_A, asst_B):
    if prompt_type == "with_persona_5":
        prompt = """Given the user profile provided below, select the response from AI assistant A or B that the user would most likely prefer. Declare your choice by using the format: "[[A]]" if you believe assistant A's response is more suitable, or "[[B]]" if assistant B's response is better suited. Additionally, assess your confidence in this decision by assigning a certainty level from 1 to 5. Use the following guidelines to assign the certainty level:

1 (Uncertain): The user profile provides insufficient information, leading to a complete lack of confidence in making a prediction.
2 (Slightly Confident): There is some evidence supporting a preference, though it is not comprehensive, and other interpretations are possible.
3 (Quite Confident): You find clear and convincing evidence that supports your prediction, though it is not entirely decisive.
4 (Confident): The user profile contains strong evidence that clearly supports your prediction, with very little ambiguity.
5 (Highly Confident): The user profile provides direct and explicit evidence that decisively supports your prediction.
Ensure you enclose your chosen certainty level in double brackets, like so: [[X]].

[User Profile]
{user_info}

[User Question]
{question}

[The Start of Assistant A's Answer]
{asst_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{asst_B}
[The End of Assistant B's Answer]

[Answer]
[[""".format(user_info=user_info, question=question, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "no_persona":
        prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{asst_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{asst_B}
[The End of Assistant B's Answer]

[Answer]
[[""".format(question=question, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "with_persona_with_tie":
        prompt = """Given the user profile provided below, select the response from AI assistant A or B that the user would most likely prefer. Declare your choice by using the format: "[[A]]" if you believe assistant A's response is more suitable, "[[B]]" if assistant B's response is better suited, or "[[C]]" for a tie.
[User Profile]
{user_info}

[User Question]
{question}

[The Start of Assistant A's Answer]
{asst_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{asst_B}
[The End of Assistant B's Answer]

[Answer]
[[""".format(user_info=user_info, question=question, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "no_confidence":
        prompt = """Given the user profile provided below, select the response from AI assistant A or B that the user would most likely prefer. Declare your choice by using the format: "[[A]]" if you believe assistant A's response is more suitable, "[[B]]" if assistant B's response is better suited.
[User Profile]
{user_info}

[User Question]
{question}

[The Start of Assistant A's Answer]
{asst_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{asst_B}
[The End of Assistant B's Answer]

[Answer]
[[""".format(user_info=user_info, question=question, asst_A=asst_A, asst_B=asst_B)
    elif prompt_type == "with_persona":
        prompt = """Given the user profile provided below, select the response from AI assistant A or B that the user would most likely prefer. Declare your choice by using the format: "[[A]]" if you believe assistant A's response is more suitable, or "[[B]]" if assistant B's response is better suited. Additionally, assess your confidence in this decision by assigning a certainty level from 1 to 100. Use the following guidelines to assign the certainty level:

1--20 (Uncertain): The user profile provides insufficient or Minimal evidence information suggests a preference. The decision is largely based on weak or indirect hints.
21--40 (Moderately Confident): There is noticeable evidence supporting a preference, though it is not comprehensive, and other interpretations are possible.
41--60 (Quite Confident): You find clear and convincing evidence that supports your prediction, though it is not entirely decisive.
61--80 (Confident): The user profile contains strong evidence that clearly supports your prediction, with very little ambiguity.
81--100 (Highly Confident): The user profile provides direct and explicit evidence that decisively supports your prediction.
Ensure you enclose your chosen certainty level in double brackets, like so: [[X]].

[User Profile]
{user_info}

[User Question]
{question}

[The Start of Assistant A's Answer]
{asst_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{asst_B}
[The End of Assistant B's Answer]

[Answer]
[[""".format(user_info=user_info, question=question, asst_A=asst_A, asst_B=asst_B)
    else:
        raise NotImplementedError
    return prompt

def get_prism_user_map(persona_features, user_data):
    user_map = {}
    if persona_features == "all_features":
        for i in range(len(user_data)):
            user_dp = user_data[i]
            user_info = []
            user_info.append(f"The person is {user_dp['gender'].lower()}")
            user_info.append(f"The person is {user_dp['age']}")
            user_info.append(f"The person is {user_dp['employment_status'].lower()}")
            user_info.append(f"The person's highest education level is {user_dp['education'].lower()}")
            user_info.append(f"The person's marital status is {user_dp['marital_status'].lower()}")
            user_info.append(f"Racially, The person is {user_dp['ethnicity']['simplified']}")
            user_info.append(f"Religiously, The person is {user_dp['religion']['simplified']}" if user_dp['religion']['simplified'] != "No Affiliation" else "The person has no religious affiliation")
            user_info.append(f"The person is born in {user_dp['location']['birth_country']} and currently lives in {user_dp['location']['reside_country']}")
            user_info = [x for x in user_info if "prefer not to say" not in x.lower()]
            user_info = ". ".join(user_info) + "."
            user_map[user_dp['user_id']] = user_info
    elif persona_features == "with_desc":
        for i in range(len(user_data)):
            user_dp = user_data[i]
            user_info = []
            user_info.append(f"The person is {user_dp['gender'].lower()}")
            user_info.append(f"The person is {user_dp['age']}")
            user_info.append(f"The person is {user_dp['employment_status'].lower()}")
            user_info.append(f"The person's highest education level is {user_dp['education'].lower()}")
            user_info.append(f"The person's marital status is {user_dp['marital_status'].lower()}")
            user_info.append(f"Racially, The person is {user_dp['ethnicity']['simplified']}")
            user_info.append(f"Religiously, The person is {user_dp['religion']['simplified']}" if user_dp['religion']['simplified'] != "No Affiliation" else "The person has no religious affiliation")
            user_info.append(f"The person is born in {user_dp['location']['birth_country']} and currently lives in {user_dp['location']['reside_country']}")
            
            
            filtered_usecases = {k: user_dp['order_lm_usecases'][k] for k in user_dp['lm_usecases'] if user_dp['lm_usecases'][k] == 1}
            sorted_usecases = sorted(filtered_usecases.items(), key=lambda x: x[1])
            usecase_string = "The person mostly use chatbot for " + ", ".join([usecase.replace('_', ' ') for usecase, order in sorted_usecases])
            user_info.append(usecase_string)
            
            pref_dict = {
                "values": "reflect his/her values or cultural perspectives",
                "creativity": "produce responses that are creative and inspiring",
                "fluency": "produce responses that are well-written and coherent",
                "factuality": "produce factual and informative responses",
                "diversity": "summarise multiple viewpoints or different worldviews",
                "safety": "produce responses that are safe and do not cause harm",
                "personalisation": "learn from our conversations and feels personalised",
                "helpfulness": "produce responses that are helpful and relevant to the requests"
            }

            sorted_prefs = sorted(user_dp['order_stated_prefs'], key=lambda x: x[1])
            sorted_prefs.remove("other")
            pref_string = "The person primarily values whether the chatbot can " + pref_dict[sorted_prefs[0]] + " and whether the chatbot can " + pref_dict[sorted_prefs[1]] 
            user_info.append(pref_string)
            # Conversely, they are less concerned with the chatbot's ability to generate creative and inspiring responses."

            user_info.append(f"The person wants AI assistant behave in the following way: {user_dp['system_string']}")
            user_info = [x for x in user_info if "prefer not to say" not in x.lower()]
            user_info = ". ".join(user_info) + "."
            user_map[user_dp['user_id']] = user_info

    elif persona_features == "key_features":
        for i in range(len(user_data)):
            user_dp = user_data[i]
            user_info = []
            user_info.append(f"The person's highest education level is {user_dp['education'].lower()}")
            user_info.append(f"Racially, The person is {user_dp['ethnicity']['simplified']}")
            user_info.append(f"The person is born in {user_dp['location']['birth_country']} and currently lives in {user_dp['location']['reside_country']}")
            
            user_info = [x for x in user_info if "prefer not to say" not in x.lower()]
            user_info = ". ".join(user_info) + "."
            user_map[user_dp['user_id']] = user_info
    elif persona_features == "least_imp_feature":
        for i in range(len(user_data)):
            user_dp = user_data[i]
            user_info = []
            
            user_info.append(f"The person is {user_dp['age']}")
            
            user_info = [x for x in user_info if "prefer not to say" not in x.lower()]
            user_info = ". ".join(user_info) + "."
            user_map[user_dp['user_id']] = user_info
    else:
        raise NotImplementedError
    
    return user_map

