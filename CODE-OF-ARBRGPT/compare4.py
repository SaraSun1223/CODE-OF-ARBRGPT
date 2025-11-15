import os
import ast
from Levenshtein import ratio
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tokenize
import io

# Compare text similarity using Levenshtein distance
def text_compare(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        code1 = f1.read()
        code2 = f2.read()
    return ratio(code1, code2)

# Simplify AST to compare only function, class, import statements
def simplify_ast(tree):
    simplified = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            simplified.append(ast.dump(node))
    return set(simplified)

# Compare AST similarity
def ast_compare(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            tree1 = ast.parse(f1.read(), filename=file1)
            tree2 = ast.parse(f2.read(), filename=file2)

        simplified1 = simplify_ast(tree1)
        simplified2 = simplify_ast(tree2)
        print("part-compare:")
        print("tree1: ", simplified1)
        print("tree2: ", simplified2)
        print("=========")
        return simplified1 == simplified2

    except SyntaxError:
        return "Syntax Error in one of the files"

def simplify_ast2(tree):
    simplified = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            simplified.append(ast.dump(node))
    return set(simplified)

# Compare AST similarity
def ast_compare2(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            tree1 = ast.parse(f1.read(), filename=file1)
            tree2 = ast.parse(f2.read(), filename=file2)

        simplified1 = simplify_ast(tree1)
        simplified2 = simplify_ast(tree2)
        # Calculate Jaccard similarity
        intersection = len(simplified1.intersection(simplified2))
        union = len(simplified1.union(simplified2))
        return intersection / union if union != 0 else 0
    except SyntaxError:
        return "Syntax Error in one of the files"



# Get tokens from code files
def get_tokens(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = list(tokenize.generate_tokens(io.StringIO(f.read()).readline))
        return [token.string for token in tokens if token.type != tokenize.COMMENT and token.string.strip()]
    except tokenize.TokenError:
        print(f"TokenError: Skipping file {file_path} due to incomplete multi-line statement.")
        return []

# Compare token similarity
def compare_tokens(file1, file2):
    tokens1 = get_tokens(file1)
    tokens2 = get_tokens(file2)

    if not tokens1 or not tokens2:
        return "N/A"

    # Convert tokens to sets for comparison
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Compare text similarity using TF-IDF + Cosine Similarity
def tfidf_cosine_similarity(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        code1 = f1.read()
        code2 = f2.read()

    vectorizer = TfidfVectorizer().fit_transform([code1, code2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Batch comparison function
def batch_compare(input_dir, output_dir, result_file):
    results = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.py'):
            input_file = os.path.join(input_dir, filename)
            print(input_file)
            output_file = os.path.join(output_dir, filename.replace('.py', '_response.py'))
            print(output_file)
            print(f"Processing file: {filename}")

            if os.path.exists(output_file):
                try:
                    text_diff = text_compare(input_file, output_file)
                    ast_sim = ast_compare(input_file, output_file)
                    ast_sim2 = ast_compare2(input_file, output_file)
                    token_sim = compare_tokens(input_file, output_file)
                    tfidf_sim = tfidf_cosine_similarity(input_file, output_file)

                    results.append({
                        'Input File': filename,
                        'Output File': os.path.basename(output_file),
                        'Text Difference': text_diff,
                        'AST Similarity': ast_sim,
                        'AST Similarity2': ast_sim2,
                        'Token Similarity': token_sim,
                        'TF-IDF Cosine Similarity': tfidf_sim,
                    })
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    results.append({
                        'Input File': filename,
                        'Output File': os.path.basename(output_file),
                        'Text Difference': 'Error',
                        'AST Similarity': 'Error',
                        'AST Similarity2': 'Error',
                        'Token Similarity': 'Error',
                        'TF-IDF Cosine Similarity': 'Error',
                    })
            else:
                results.append({
                    'Input File': filename,
                    'Output File': os.path.basename(output_file),
                    'Text Difference': 'Output file does not exist',
                    'AST Similarity': 'N/A',
                    'AST Similarity2': 'N/A',
                    'Token Similarity': 'N/A',
                    'TF-IDF Cosine Similarity': 'N/A',
                })

    df = pd.DataFrame(results)
    df.to_excel(result_file, index=False)

if __name__ == "__main__":
    for framework in ['PyTorch', 'TensorFlow', 'MXNet']:
        # for method in ['output_OS_KW_S2R_CoT',
        #                'output_OS_S2R_CoT',
        #                'output_OS_KW_S2R',
        #                'output_OS',
        #                ''output_KW_S2R_CoT'',
        #                ]:
        # for method in ['llmtest', 'canllm']:
        # for method in ['deepseek_output_OS_KW_S2R_CoT', 'deepseek_output_OS_S2R_CoT', 'deepseek_output_KW_S2R_CoT',
        #                'deepseek_output_OS_KW_S2R', 'deepseek_output_OS']:
        # for method in ['qwen_output_OS_KW_S2R_CoT', 'qwen_output_OS_S2R_CoT', 'qwen_output_KW_S2R_CoT',
        #                'qwen_output_OS_KW_S2R', 'qwen_output_OS']:
            # for method in ['deepseek_output_OS_S2R_CoT']:
        # for method in ['qwenP_output_OS_KW_S2R_CoT']:
        for method in ['deepseek_output_llmtest', 'deepseek_output_canllm']:
        # for method in ['qwen_output_llmtest','qwen_output_canllm']:

            parent = f'./result/{framework}'
            os.makedirs(parent, exist_ok=True)

            # input_folder = f'./input/{framework}_bugreports/code'
            input_folder = f'./code/{framework}'

            # output_folder = './out/code_canllm'
            # output_folder = './out/code_llmfew'
            output_folder = f'./out/{framework}/{method}/code'
            # output_folder = './out/output_OS_S2R_CoT/code'
            # output_folder = './out/output_OS_KW_S2R/code'
            # output_folder = './out/output_OS_S2R/code'
            # output_folder = './out/output_OS5/code'
            # output_folder = './out/output_5/code'
            # result_excel = 'comparison_results_canllm.xlsx'
            # result_excel = 'comparison_results_llmfew.xlsx'
            result_excel = f'./result/{framework}/{method}_all.xlsx'
            # result_excel = './result/output_OS_S2R_CoT1.xlsx'
            # result_excel = './result/output_OS_KW_S2R1.xlsx'
            # result_excel = './result/output_OS_S2R1.xlsx'
            # result_excel = './result/output_OS5.xlsx'
            # result_excel = './result/output_5.xlsx'
            batch_compare(input_folder, output_folder, result_excel)
