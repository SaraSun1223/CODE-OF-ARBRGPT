import os
import openai

KEY = os.getenv('OPENAI_API_KEY')

def get_chat_messages(prompt_text):
    openai.api_key = KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_text,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.3,
    )
    return response.choices[0].message.content


def read_files_from_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    return files


def process_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = read_files_from_directory(input_directory)
    for file_name in files:
        with open(os.path.join(input_directory, file_name), 'r', encoding='utf-8') as file:
            bug_report = file.read()
            bug_id = file_name.split('_')[0]

        # 定义对话信息
        action_prompt = "Available actions: [Install, Configure, Train, Monitor, Debug, Optimize, Save, Load, Test, Report]"
        object_prompt = "Available objects: [TensorFlow, Dataset, Model, Session, GPU Memory, Environment, Error Log, Configuration File, Checkpoint, TFRecord File]"
        roses_prompt = """
Step1-Step to reproduce, I want you to use keywords to describe the step to reproduce, and use the format of [Available actions] (Available objects). You must explain your thinking and reasoning process step by step. 
Step2-requirements and reproduce code, output a self-contained example that can reproduce the issue, including a list of the environments needed to reproduce the code and the minimal reproducible test case.
Step3-Dockerfile, output the DockerFile that replicates the desired environment.  
Step4-running result, output the defect phenomenon in the format as follows.      
[symptom](Available Option: Build Failure, Crash, Hang, Incorrect Functionality, Poor Performance, Unreported)
[root_cause](Available Option: API Incompatibility, API Misuse, Concurrency, Dependent Module Issue, Environment Incompatibility, Incorrect Algorithm Implementation, Incorrect Assignment, Incorrect Exception Handling, Misconfiguration, Numerical Issue, Others, Tensor Shape Misalignment, Type Confusion)	
[component](Available Option: Environment, General Utility Implementation, Graph-Level Implementation, Operation Implementation, User-Level API)	
[stage](Available Option: deployment, install, preprocessing,training, utility)
You need to make sure that your output code is a complete and runnable file, please do not have syntax errors.
You should answer the question in exactly the following format(bug_id should be replace by a number):
{Step to reproduce}
{requirements.txt}
{bug_id.py}
{Dockerfile}
{running result}
        """



        question_prompt = """Question: 
Bug Report: {}
Bug ID: {}
        """.format(bug_report, bug_id)


        prompt_text = [{"role": "system",
                        "content": "You are a senior bug report expert.Please remember your duty is to understand the bug report and identifying the steps to reproduce the issue."},
                       {"role": "user",
                        "content": action_prompt + "\n" + object_prompt + "\n" + roses_prompt},
                       {"role": "assistant",
                        "content": "I see. I will generate the answer in formatted string -> {Step to reproduce}\n{requirements.txt}\n{bug_id.py}\n{running result}"},
                       {"role": "user",
                        "content": question_prompt}
                       ]

        response_content = get_chat_messages(prompt_text)
        # 打印生成的回复
        print("ChatGPT：", response_content)
        output_file_name = os.path.splitext(file_name)[0] + '_response.txt'
        with open(os.path.join(output_directory, output_file_name), 'w', encoding='utf-8') as output_file:
            output_file.write(response_content)


# 使用示例
for framework in ['PyTorch', 'TensorFlow', 'MXNet']:
    input_directory = f'./input/{framework}_bugreports/summary_with_code'
    output_directory = (f'./output/{framework}/output_KW_S2R_CoT')
    process_files(input_directory, output_directory)
# framework = 'TensorFlow'
# input_directory = f'./input/{framework}_bugreports/summary_with_code'
# output_directory = (f'./output/{framework}/output_OS_KW_S2R_CoT1')
# process_files(input_directory, output_directory)


