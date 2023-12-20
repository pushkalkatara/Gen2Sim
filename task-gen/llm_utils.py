import os
import openai

# Use this for custom configs

# openai.api_type = "azure"
# openai.api_base = "https://katfgroup-gpt4-ce.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ["OPENAI_API_KEY"]

def query(prompt, log_dir):
    path_prompts = os.path.join(log_dir, "prompts")
    if not os.path.exists(path_prompts):
        os.mkdir(path_prompts)

    try:
        response = openai.ChatCompletion.create(
            engine="gpt-4",
            messages = [{"role": "system", "content": prompt}],
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        response = response.choices[0].message.content
        with open(f"{path_prompts}/response.txt", "w") as text_file:
            text_file.write(response)
        with open(f"{path_prompts}/prompt.txt", "w") as text_file:
            text_file.write(prompt)
        return response
    except:
        with open(f"{path_prompts}/prompt_failed.txt", "w") as text_file:
            text_file.write(prompt)
        return None