import os
import openai

'''
# Use this for custom configs
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""
'''

def query(prompt, log_dir, cls_id, name):

    path_responses = os.path.join(log_dir + "prompts_responses")
    if not os.path.exists(path_responses):
        os.mkdir(path_responses)

    path_failed = os.path.join(log_dir + "failed_prompts")
    if not os.path.exists(path_responses):
        os.mkdir(path_failed)
    
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-4",
            messages = [{"role": "system", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        response = response.choices[0].message.content

        with open(f"{path_responses}/response_{cls_id}_{name}.txt", "w") as text_file:
            text_file.write(response)

        with open(f"{path_responses}/prompt_{cls_id}_{name}.txt", "w") as text_file:
            text_file.write(prompt)

        return response

    except:
        with open(f"{path_failed}/prompt_{cls_id}_{name}.txt", "w") as text_file:
            text_file.write(prompt)
        return None