import os
import re

def split_file_content(file_path, framework):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 提取文件编号
    file_number = os.path.basename(file_path).split('_')[0]

    # 使用正则表达式匹配各个部分的内容
    step_to_reproduce_match = re.search(r'\{Step to reproduce\}(.*?)\{requirements\.txt\}', content, re.DOTALL)
    # requirements_match = re.search(r'\{requirements\.txt\}(.*?)\{' + re.escape(file_number) + r'\.py\}', content, re.DOTALL)
    requirements_match = re.search(r'\{requirements\.txt\}(.*?)\{', content, re.DOTALL)
    # requirements_match = re.search(r'\{requirements\.txt\}(.*?)\{bug_id\.py\}', content, re.DOTALL)

    # code_match = re.search(r'\{' + re.escape(file_number) + r'\.py\}(.*?)\-', content, re.DOTALL)
    code_match = re.search(r'\{' + re.escape(file_number) + r'\.py\}(.*?)\{Dockerfile\}', content, re.DOTALL)
    if code_match is None:
        code_match = re.search(r'\.py\}(.*?)\{Dockerfile\}', content, re.DOTALL)
    if code_match is None:
        code_match = re.search(r'```python(.*?)```', content, re.DOTALL)
    # code_match = re.search(r'\{bug_id\.py\}(.*?)\{Dockerfile\}', content, re.DOTALL)
    docker_match = re.search(r'\{Dockerfile\}(.*?)\{running result\}', content, re.DOTALL)
    running_result_match = re.search(r'\{running result\}(.*?)$', content, re.DOTALL)

    # if not step_to_reproduce_match:
    #     print(f"Error: {file_path} - Step to reproduce not found")
    #     return
    if not requirements_match:
        print(f"Error: {file_path} - requirements.txt not found")
        return
    if not code_match:
        print(f"Error: {file_path} - {file_number}.py not found")
        return
    if not docker_match:
        print(f"Error: {file_path} - docker file not found")
        return
    if not running_result_match:
        print(f"Error: {file_path} - running result not found")
        return

    # step_to_reproduce = step_to_reproduce_match.group(1).strip()
    requirements = requirements_match.group(1).strip()
    code = code_match.group(1).strip()
    if '```python' in code:
        code = code.replace("```python", '')
        code = code.replace("```", '')
    docker = docker_match.group(1).strip()
    if '```Dockerfile' in docker:
        docker = docker.replace("```Dockerfile", '')
        docker = docker.replace("```", '')
    running_result = running_result_match.group(1).strip()

    # 定义目标文件夹
    file_output_folder= f"./out/{framework}/{method}/{file_number}"
    # file_output_folder= f"./out/output_OS_S2R_CoT5/{file_number}"
    # file_output_folder= f"./out/output_OS_KW_S2R5/{file_number}"
    # file_output_folder= f"./out/output_OS_S2R5/{file_number}"
    # file_output_folder= f"./out/output_OS5/{file_number}"
    # file_output_folder= f"./out/output_5/{file_number}"

    # s2r_folder = './out/{method}/S2R'
    # requirements_folder = './out/{method}/requirements'

    code_folder = f'./out/{framework}/{method}/code'
    # code_folder = './out/output_OS_S2R_CoT5/code'
    # code_folder = './out/output_OS_KW_S2R5/code'
    # code_folder = './out/output_OS_S2R5/code'
    # code_folder = './out/output_OS5/code'
    # code_folder = './out/output_5/code'

    # results_folder = './out/{method}/results'
    #
    # # 创建文件夹（如果不存在）
    os.makedirs(file_output_folder, exist_ok=True)
    # os.makedirs(s2r_folder, exist_ok=True)
    # os.makedirs(requirements_folder, exist_ok=True)
    os.makedirs(code_folder, exist_ok=True)
    # os.makedirs(results_folder, exist_ok=True)

    # 写入到不同的文件中
    # with open(os.path.join(file_output_folder, f'{file_number}_S2R.txt'), 'w', encoding='utf-8') as s2r_file:
    #     s2r_file.write(step_to_reproduce)

    with open(os.path.join(file_output_folder, f'requirements.txt'), 'w', encoding='utf-8') as requirements_file:
        requirements_file.write(requirements)

    with open(os.path.join(code_folder, f'{file_number}_response.py'), 'w', encoding='utf-8') as response_file:
        response_file.write(code)
    with open(os.path.join(file_output_folder, f'{file_number}.py'), 'w', encoding='utf-8') as response_file:
        response_file.write(code)

    with open(os.path.join(file_output_folder, f'Dockerfile'), 'w', encoding='utf-8') as docker_file:
        docker_file.write(docker)

    with open(os.path.join(file_output_folder, f'{file_number}_result.txt'), 'w', encoding='utf-8') as result_file:
        result_file.write(running_result)

def process_all_files_in_folder(folder_path,framework):
    for filename in os.listdir(folder_path):
        if filename.endswith('_summary_with_code_response.txt'):
            file_path = os.path.join(folder_path, filename)
            split_file_content(file_path, framework)

for framework in ['PyTorch', 'TensorFlow', 'MXNet']:
    # for method in ['output_OS_KW_S2R_CoT','output_OS_S2R_CoT', 'output_OS_KW_S2R', 'output_OS']:
    # for method in ['output_KW_S2R_CoT']:
    # for method in ['deepseek_output_OS_KW_S2R_CoT','deepseek_output_OS_S2R_CoT','deepseek_output_KW_S2R_CoT','deepseek_output_OS_KW_S2R','deepseek_output_OS']:
    for method in ['qwen_output_OS_KW_S2R_CoT','qwen_output_OS_S2R_CoT','qwen_output_KW_S2R_CoT','qwen_output_OS_KW_S2R','qwen_output_OS']:



        # 使用示例
        folder_path = f'./output/{framework}/{method}'
        # folder_path = './output/output_OS_S2R_CoT5'
        # folder_path = './output/output_OS_KW_S2R5'
        # folder_path = './output/output_OS_S2R5'
        # folder_path = './output/output_OS5'
        # folder_path = './output/output_5'

        process_all_files_in_folder(folder_path,framework)
