import os
from openai import OpenAI

# os.environ["DASHSCOPE_API_KEY"] = ""os.getenv("DASHSCOPE_API_KEY")

KEY = os.getenv("DASHSCOPE_API_KEY")

def get_chat_messages(prompt_text):
    client = OpenAI(
        api_key=KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=prompt_text,
        extra_body={"enable_thinking": False},
        temperature=0.3,
        max_tokens=2000
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
        question_prompt = """Write a Python reproduce case for the following bug report: 
Bug Report: {}
Bug ID: {}
            """.format(bug_report, bug_id)

        prompt_text = [{"role": "system",
                        "content": "You are a senior bug report expert.Please remember your duty is to understand the bug report and reproduce the issue."},
                       {"role": "user",
                        "content": question_prompt}
                       ]

        response_content = get_chat_messages(prompt_text)
        # 打印生成的回复
        print("ChatGPT：", response_content)
        output_file_name = os.path.splitext(file_name)[0] + '_response.txt'
        with open(os.path.join(output_directory, output_file_name), 'w', encoding='utf-8') as output_file:
            output_file.write(response_content)

# 'PyTorch', 'TensorFlow',
# 使用示例
for framework in ['MXNet']:
    input_directory = f'./input/{framework}_bugreports/summary_with_code'
    output_directory = (f'./output/{framework}/deepseek_output_canllm')
    process_files(input_directory, output_directory)



