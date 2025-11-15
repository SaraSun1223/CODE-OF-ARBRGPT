import os
import openai


KEY = os.environ.get('OPENAI_API_KEY')
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

        question_prompt = """Provide a self-contained example that reproduces this issue:  
        Bug Report: {}
        Bug ID: {}
        """.format(bug_report, bug_id)

        prompt_text = [{"role": "system",
                        "content": "You are a senior bug report expert."},
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
for framework in ['PyTorch', 'TensorFlow', 'MXNET']:
    for method in ['llmtest']:
        input_directory = f'./input/{framework}_bugreports/summary_with_code'
        output_directory = (f'./output/{framework}/{method}')
        process_files(input_directory, output_directory)
# framework = 'TensorFlow'
# input_directory = f'./input/{framework}_bugreports/summary_with_code'
# output_directory = (f'./output/{framework}/output_OS')
# process_files(input_directory, output_directory)


