# main.py
import os
import subprocess


def run_part1():
    """执行第一部分：生成缺陷复现步骤和代码"""
    from chatGPT3_5_OS_KW_S2R_CoT import process_files
    process_files('./input/TensorFlow_bugreports/summary_with_code', './output/TensorFlow1/output_OS_KW_S2R_CoT')


def run_part2():
    """执行第二部分：处理响应并执行环境搭建"""
    # 分割响应文件
    from split_response import process_all_files_in_folder
    process_all_files_in_folder('./output/TensorFlow/output_OS_KW_S2R_CoT')

    # # 执行环境搭建和复现
    # from test_closure import main as run_test_closure
    # run_test_closure()


def main():
    # 创建必要目录
    # os.makedirs('./input', exist_ok=True)
    # os.makedirs('./output/output_OS_KW_S2R_CoT', exist_ok=True)

    print("======== 开始执行缺陷处理流程 ========")

    print("\n>>> 正在执行第一部分：生成复现步骤和代码...")
    run_part1()

    print("\n>>> 正在执行第二部分：处理响应并执行复现...")
    run_part2()

    print("\n======== 流程执行完成 ========")


if __name__ == "__main__":
    main()