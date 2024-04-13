from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, \
      StoppingCriteriaList, LogitsProcessorList, LogitsProcessor
from tqdm import tqdm
import time
import os
import tempfile
import subprocess
from subprocess import CompletedProcess
import sys
from io import StringIO
from utils import REPOS_DIR, load_jsonl, dump_jsonl, MODEL_VECTOR_SIZE
from analysize import analysize
import re
import torch
from eclipse_jdt_ls import get_import_src_for_impt_stmt

from pylint.lint import Run
from pylint.reporters.text import TextReporter
from pylint.lint import pylinter
from typing import List

import json

def add_sys_path(file_path, repo_name, repo_dir):
    added_path = []
    filepath = file_path
    added_path.append(os.path.dirname(filepath))

    sub_repo_dir = repo_dir + "/" + repo_name
    added_path.append(sub_repo_dir)
    added_path.append(sub_repo_dir + "/src")
    added_path.append(sub_repo_dir + "/" + repo_name)

    added_path.append(os.path.dirname(os.path.dirname(filepath)))

    src_index = filepath.find("src")
    if src_index != -1:
        added_path.append(filepath[:src_index])
        added_path.append(filepath[:src_index+3])

    test_index = filepath.find("tests")
    if test_index != -1:
        added_path.append(filepath[:test_index])

    app_index = filepath.find("app")
    if app_index != -1:
        added_path.append(filepath[:app_index])
        added_path.append(filepath[:app_index+3])

    added_path.append(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))
    added_path = [path for path in added_path if os.path.exists(path)]
    for path in added_path:
        if not path in sys.path:
            sys.path.append(path)
    return added_path


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

def get_symbol_embeddings(model_name):
    model_vector_size = MODEL_VECTOR_SIZE[model_name]
    symbol_probability = torch.zeros(model_vector_size)
    symbols = []
    newline_token = []
    with open("python-symbol.json", "r") as f:
        seps = json.load(f)
    for i in range(model_vector_size):
        token =  tokenizer.decode([i], skip_special_tokens = True)
        if token and token[0] in seps:
            symbols.append(i)
        if token.startswith("\n"):
            newline_token.append(i)
    for sidx in symbols:
        symbol_probability[sidx] = 1
    return symbol_probability, newline_token

def init_model(model_id, device):
    if model_id == "starcoder":
        model_id = "bigcode/starcoderbase"
    elif model_id == "deepseek-coder":
        model_id = "data/models/deepseek-coder-6.7b-base"
    elif model_id == "codellama":
        model_id = "codellama/CodeLlama-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return model, tokenizer


def check_syntax(code, file_path, sub_repo_dir, lineno):
    current_directory = os.getcwd()
    if current_directory in sys.path:
        sys.path.remove(current_directory)
    add_sys_path(file_path, sub_repo_dir, REPOS_DIR["python"])
    new_file_path = re.sub(r".py$", "_tmp.py", file_path)
    # new_file_path = file_path.replace(".py", "_tmp.py")
    if os.path.exists(new_file_path):
        timestamp = int(time.time())
        new_file_path = file_path.replace(".py", f"_{timestamp}.py")
    with open(new_file_path, "w") as f:
        f.write(code)
    pylint_output = StringIO()  # Custom open stream
    reporter = TextReporter(pylint_output)
    pylinter.MANAGER.clear_cache()
    Run([new_file_path], reporter=reporter, exit=False)
    res = pylint_output.getvalue()
    pattern = fr':{lineno}:[0-9]*: ([A-Z][0-9]{{4}}):'
    for line in res.split("\n"):
        match = re.findall(pattern, line)
        if match and match[0][0] == 'E' and not "Module 'torch'" in line:
            os.remove(new_file_path)
            return False , line
    os.remove(new_file_path)
    return True, ""

# ----------- java --------------

def run_javac_cmd(target_fpath: str, deps_fpath: List[str]) -> CompletedProcess:

    command = ['javac', target_fpath] + list(deps_fpath) + ['-Xmaxerrs', '1000']
    result = subprocess.run(command, capture_output=True, text=True)
    # print("Return code:", result.returncode)
    # print("Standard output:", result.stdout)
    # print("Standard error:", result.stderr)
    return result

def analyze_javac_error_info(fpath: str, lineno: int, result_str: str) -> List[str]:
    errors = []
    key = f'{fpath}:{lineno}'
    for line in result_str.split('\n'):
        if key not in line: continue
        idx = line.find('error: ')
        if line.find('error: ') != -1:
            errors.append(line[idx:])
    return errors

def pre_javac_check(datas):
    repos_dir = REPOS_DIR['java']
    pre_javac_errs = {}
    for data in tqdm(datas):
        
        fpath = os.path.join(repos_dir, data['metadata']['repository'], data['metadata']['file'])
        import_definitions = get_import_src_for_impt_stmt(fpath)
        result = run_javac_cmd(fpath, import_definitions.values())
        if len(result.stderr.strip()) != 0:
            gt_errs = analyze_javac_error_info(fpath, data['metadata']['groundtruth_start_lineno'] + 1, result.stderr)
            pre_javac_errs[data['metadata']['task_id']] = gt_errs
        data['errors'] = gt_errs
    return pre_javac_errs

def load_java_pre_errs(err_fpath = 'data/datasets/cceval/java/errs.json') -> dict:
    with open(err_fpath, 'r') as f:
        err_dict = json.load(f)
    return err_dict

def generate_random_string(length):  
    import random  
    import string  
    letters_and_digits = string.ascii_letters + string.digits  
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def java_check_syntax(gen_line, file_path, lineno, task_id):
    err_dict = load_java_pre_errs()

    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines[lineno - 1] = gen_line
    gen_src = ''.join(lines)

    temp_fpath = os.path.join(os.path.dirname(file_path), f'{generate_random_string(8)}.java')
    with open(temp_fpath, 'w') as f:
        f.write(gen_src)
        
    try:
        import_definitions = get_import_src_for_impt_stmt(file_path)
        result = run_javac_cmd(temp_fpath, import_definitions.values())
        if len(result.stderr.strip()) != 0:
            gen_errs = analyze_javac_error_info(temp_fpath, lineno, result.stderr)
        else:
            return True, ""
        
        if task_id not in err_dict.keys() or len(gen_errs) != len(err_dict[task_id]):
            return False, gen_errs
        
        for err in gen_errs:
            if err not in err_dict[task_id]:
                return False, gen_errs
        return True, ""
    
    finally:
        os.remove(temp_fpath)
 
if __name__ == "__main__":
    device = "cuda:0"
    model, tokenizer =init_model("deepseek-coder", device)
    data = load_jsonl("data/datasets/cceval/cceval_2460.jsonl")

    stopping_criteria = StoppingCriteriaList()
    symbol_embeddings, newline_token_list = get_symbol_embeddings("deepseek-coder")
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=newline_token_list))

    with torch.no_grad():
        for case in tqdm(data[699:]):
            filepath = REPOS_DIR + "/"  + case["metadata"]["repository"] + "/" + case["metadata"]["file"]
            # add_sys_path(filepath, case["metadata"]["repository"], REPO_DIR)
            # print(sys.path)
            
            input_ids = tokenizer(case["prompt"], return_tensors="pt").input_ids.to(device)
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens = 128,
                stopping_criteria = stopping_criteria,
                num_return_sequences=3
            )

            has_valid_output = False

            outputs = []
            for d in generated_ids:
                outputs.append(tokenizer.decode(d[len(input_ids[0]):]).strip().split("\n")[0])
            
            case["error"] = {}
            for idx, output in enumerate(outputs, 1):
                code = case["metadata"]["prefix_src"] + case["prompt"] + output
                without_error, error_message = check_syntax(code, filepath, case["metadata"]["groundtruth_start_lineno"]+1)
                if without_error:
                    has_valid_output = True
                    case["predict"] = output
                    break
                else:
                    case["error"][idx] = error_message
            if not has_valid_output:
                case["predict"] = outputs[0]
            
            print(case["prompt"].split("\n")[-1] + case["predict"])
            print(case["error"])
            input()