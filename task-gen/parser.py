import re


CLASS_TEMPLATE = '''
class {}(BaseTask):
    def __init__(
        self,
        env,
        active_env_state
    ):
        super().__init__(
            env=env,
            active_env_state=active_env_state
        )

        self.task_desc = "{}"

{}
'''

def indent_string(input_string, indent_level):
    lines = input_string.split('\n')
    indented_lines = [(indent_level * '    ') + line for line in lines]
    indented_string = '\n'.join(indented_lines)
    return indented_string

def generate_task(task_name, task_desc, code, dest_class):
    code = indent_string(code, 1)
    new_code = CLASS_TEMPLATE.format(task_name, task_desc, code)
    with open(dest_class, "a") as f:
        f.write(new_code)

def parse_tasks(gpt_response, dest_class):
    tasks = []
    pattern = re.compile(r'Task: (.+?)\nTask Description: "(.+?)"\n```(.+?)```', re.DOTALL)
    matches = pattern.findall(gpt_response)
    for match in matches:
        name, description, code = match
        generate_task(name.strip(), description.strip(), code.strip(), dest_class)
        tasks.append(name.strip())
    return tasks

if __name__ == '__main__':
    file_path = '/projects/katefgroup/learning-simulations/log_dir/task-gen-log/prompts/response.txt'
    with open(file_path, 'r') as file:
        response = file.read()

    parse_tasks(response, '/home/pkatara/learning-simulations/gym/envs/gpt_task_generated.py')
