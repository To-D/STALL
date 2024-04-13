
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import json
from typing import List
import traceback

import torch
import asyncio
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, \
      StoppingCriteriaList, LogitsProcessorList, LogitsProcessor

from utils import load_jsonl, dump_jsonl, REPOS_DIR, MODEL_VECTOR_SIZE
from prompt_builder import PromptBuilder
from evaluate import evaluate
from eclipse_jdt_ls import single_request, REQUEST_TYPE
from feature import is_API_invocation_or_completion
from postprocessing import java_check_syntax

from parso.cache import clear_cache, _get_default_cache_path
import sys
sys.path.append('PriorWorks/monitors4codegen-lsp/src')
sys.path.append('PriorWorks/monitors4codegen-lsp')

from monitors4codegen.multilspy.language_server import SyncLanguageServer
from monitors4codegen.multilspy.multilspy_logger import MultilspyLogger
from monitors4codegen.multilspy.multilspy_config import Language, MultilspyConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from monitors4codegen.monitor_guided_decoding.monitors.dereferences_monitor import DereferencesMonitor
from monitors4codegen.monitor_guided_decoding.monitor import MonitorFileBuffer
from monitors4codegen.monitor_guided_decoding.hf_gen import MGDLogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from monitors4codegen.monitor_guided_decoding.tokenizer_wrapper import HFTokenizerWrapper

def clear_sa_cache():
    cache_path = _get_default_cache_path()
    if not os.path.exists(cache_path):
        try:
            os.mkdir(cache_path)
            clear_cache()
        except:
            return

def init_model(model_id, device):#, using_postprocessing=False):
    if model_id == "starcoder":
        model_id = "bigcode/starcoderbase-7b"
    elif model_id == "deepseek-coder":
        model_id = "data//models/deepseek-coder-6.7b-base"
    elif model_id == "codellama":
        model_id = "codellama/CodeLlama-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    return model, tokenizer

def get_symbol_embeddings():
    symbol_probability = torch.zeros(MODEL_VECTOR_SIZE)
    symbols = []
    newline_token = []
    with open("python-symbol.json", "r") as f:
        seps = json.load(f)
    for i in range(MODEL_VECTOR_SIZE):
        token =  tokenizer.decode([i], skip_special_tokens = True)
        if token and token[0] in seps:
            symbols.append(i)
        if '\n' in token:
            newline_token.append(i)
    for sidx in symbols:
        symbol_probability[sidx] = 1
    return symbol_probability, newline_token

def extract_last_identifier(java_code):
    identifier = ""
    i = len(java_code) - 1
    
    # 从字符串末尾往前遍历
    while i >= 0:
        if java_code[i].isalnum() or java_code[i] == '_' or java_code[i].isalpha():
            identifier = java_code[i] + identifier
        else:
            break
        i -= 1
    
    if len(identifier) != 0 and identifier[0].isdigit():
        raise Exception('fail to extract last identifier.')
    return identifier

def static_analysis(query, symbol_embeddings, so_far_gen):
    # options = analysize(REPOS_DIR["java"], query)
    
    repo = query['metadata']['repository']
    fpath = query['metadata']['file']
    line = query['metadata']['groundtruth_start_lineno']
    col = query['metadata']['groundtruth_start_colno'] + 1
    options = asyncio.run(single_request(os.path.join(REPOS_DIR["java"], repo), fpath, line, col, REQUEST_TYPE.COMPLETION))

    if options:
        analysis_probobility = torch.zeros(MODEL_VECTOR_SIZE)
        for option in options:
            # java中只返回完整的completionText
            if query['prompt'].endswith(option['completionText']):
                # TODO: symbo_embddings
                # 一个补全补完后，下一个可能是符号
                for idx, i in enumerate(symbol_embeddings):
                    analysis_probobility[idx] = i
                prefix = option['completionText']
                # 也有可能是以该补全为前缀的另一个补全
                for option in options:
                    if option['completionText'].startswith(prefix):
                        next_token_id = tokenizer(option['completionText'], add_special_tokens = False).input_ids[0]
                        analysis_probobility[next_token_id] = 1
                break
            # 有大小写相同的情况，如AAA和aaa，都会返回''
            if option['completionText'] == '': continue
            last_identifier = extract_last_identifier(so_far_gen)
            complete_suffix = option['completionText'][len(last_identifier):]
            next_token_ids = tokenizer(complete_suffix, add_special_tokens = False).input_ids
            if next_token_ids:
                next_token_id = next_token_ids[0]
                analysis_probobility[next_token_id] = 1
        return analysis_probobility, options
    else:
        return None, []

class SALogitsProcessor(LogitsProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.so_far_gen = ''

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        query = self.query
        query["output"][self.turn] = {}
        if len(input_ids[0]) > query["input_len"]:
            next_token = tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
            query["prompt"] += next_token
            query['metadata']["groundtruth_start_colno"] += len(next_token)
            query["output"][self.turn-1]["predict"] = next_token
            self.so_far_gen += next_token

        self.turn += 1

        if not is_API_invocation_or_completion(query["prompt"]):
            return scores

        sa_prob, options = static_analysis(query, self.symbol_embeddings, self.so_far_gen)
        query["output"][self.turn-1]["options"] = options

        if len(options)==0:
            return scores

        scores.to(device)
        sa_prob.to(device)
        scores[0][sa_prob == 0] = float('-inf')
        return scores

    def set_turn(self, turn):
        self.turn = turn

    def set_query(self, query):
        self.query = query

    def set_symbol_embeddings(self, symbol_embeddings):
        self.symbol_embeddings = symbol_embeddings

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list
        self.stop_conditions = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.stop_conditions == None:
            self.stop_conditions = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            if self.stop_conditions[i] == False:
                self.stop_conditions[i] = input_ids[i][-1].detach().cpu().numpy() in self.token_id_list
        if self.stop_conditions.int().sum().item() == input_ids.shape[0]:# all(self.stop_conditions):
            self.stop_conditions.fill_(False)
            return True
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help="model id")
    type_info = """
    'i': infile context only
    'r': + retrival context
    'si': + static analysis context through import statements
    'sls': + static analysis context through language server
    'sd': + static analysis when decoding
    'sp': + static analysis when filtering the output
    'si-d': combine si and sd
    'si-p': combine si and sp
    'sls-d': combine sls and sd
    'sls-p': combiner sls and sp
    'r-si': omit
    'r-sls': omit
    'r-sd': omit
    'r-sp': omit
    'r-si-d': omit
    'r-si-p': omit
    'r-sls-d': omit
    'r-sls-p': omit
    'si-d-p': 
    'r-si-d-p':
    """
    parser.add_argument('-t', '--type',
                        required=True,
                        choices=["i","r", "si", "sls", "sd", "sp", "si-d", "si-p", "sls-d", "sls-p",
                                  "r-si", "r-sls", "r-sd", "r-sp", "r-si-d", "r-si-p", "r-sls-d", "r-sls-p", "si-d-p", "r-si-d-p"],
                        help=type_info)
    parser.add_argument('-i', '--input', type=str, help="input file")
    parser.add_argument('-g', '--gpu', type=int, help="gpu id")
    parser.add_argument('-cvd', '--cuda_visible_devices', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('-l', '--language', type=str, default='python', help="target language")

    return parser.parse_args()

def build_prompt(data, prompt_type):
    builder = PromptBuilder(prompt_type)
    builder.build(data)

def load_data(args):
    src_data =load_jsonl(args.input)
    prompt_file_path = args.input.replace(".jsonl", f"-{args.type}.jsonl")
    if os.path.exists(prompt_file_path):
        src_data = load_jsonl(prompt_file_path)
    else:
        print("Constructing prompt...")
        build_prompt(src_data,args.type)
        dump_jsonl(src_data, prompt_file_path)
    return src_data

def sava_oom_errs(dir, filename, errors):
    with open(os.path.join(dir, filename + '.txt'), 'w') as f:
        for err in errors:
            f.write(err)
            f.write('\n')

def generate_with_post_processing(model, tokenizer, input_ids, query, stopping_criteria, language, oom_task_id, logits_processor=None):
    # try:
    #     if logits_processor == None:
    #         generated_ids = model.generate(
    #             input_ids,
    #             pad_token_id = tokenizer.eos_token_id,
    #             max_new_tokens = 64,
    #             stopping_criteria = stopping_criteria,
    #             num_beams=3,
    #             num_return_sequences=3
    #         )
    #     else:
    #         generated_ids = model.generate(
    #             input_ids,
    #             pad_token_id = tokenizer.eos_token_id,
    #             max_new_tokens = 64,
    #             logits_processor = logits_processor,
    #             stopping_criteria = stopping_criteria,
    #             num_beams=3,
    #             num_return_sequences=3
    #         )


    #     has_valid_output = False
    #     outputs = []
    #     for i in range(generated_ids.shape[0]):
    #         outputs.append(tokenizer.decode(generated_ids[i][len(input_ids[0]):], skip_special_tokens=True).strip().split("\n")[0])
        
    #     query["error"] = {}
    #     for idx, output in enumerate(outputs, 1):
    #         # code = query["metadata"]["prefix_src"] + query["prompt"] + output
    #         gen_line = query["prompt"].split('\n')[-1] + output
    #         filepath = REPOS_DIR[language] + "/"  + query["metadata"]["repository"] + "/" + query["metadata"]["file"]
    #         without_error, error_message = java_check_syntax(gen_line, filepath, query["metadata"]["groundtruth_start_lineno"] + 1, query["metadata"]["task_id"])
    #         if without_error:
    #             has_valid_output = True
    #             query["predict"] = output
    #             break
    #         else:
    #             query["error"][idx] = {"predict":output ,"err_message":error_message}
    #     if not has_valid_output:
    #         query["predict"] = outputs[0]
    #     return True
        
    # except Exception as e:
    #     # print('generate_with_post_processing')
    #     # print(e)
    #     # traceback.print_exc()
    #     oom_task_id.append(query['metadata']['task_id'])
    #     if logits_processor != None:
    #         return False
    try:
        generated_ids = model.generate(
            input_ids,
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens = 64,
            stopping_criteria = stopping_criteria,
        )
        query['predict'] = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens = True).strip()
        return True
    except:
        print('greedy oom:', query["metadata"]["task_id"])
        return False

def predict(model, tokenizer, data, output_file, using_decoding=False, using_postprocessing=False, language='java'):
    has_predicted = []
    if os.path.exists(output_file):
        for line in load_jsonl(output_file):
            has_predicted.append(line["metadata"]["task_id"])

    stopping_criteria = StoppingCriteriaList()
    symbol_embeddings, newline_token_list = get_symbol_embeddings()
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=newline_token_list))
    oom_task_id = []

    if using_decoding:
        for i, query in enumerate(tqdm(data), 1):
            if query["metadata"]["task_id"] in has_predicted:
                continue
            task_repo_path = os.path.join(REPOS_DIR[language], query['metadata']['repository'])
            filepath = query['metadata']['file']
            cursor_pos = (query['metadata']['groundtruth_start_lineno'], query['metadata']['groundtruth_start_colno'])
            logger = MultilspyLogger()
            logger.logger.setLevel(logging.ERROR)
            lsp = SyncLanguageServer.create(MultilspyConfig(language), logger, task_repo_path)
            success_start = False
            sucesss_d_p = False
            while not success_start:
                try:
                    with lsp.start_server():
                        with lsp.open_file(filepath):
                            success_start = True
                            filebuffer = MonitorFileBuffer(lsp.language_server, filepath, cursor_pos, cursor_pos, language, "")
                            
                            if using_postprocessing:
                                dereference_monitors = []
                                # BeamSearch的长度是3，所以生成三个monitor
                                for _ in range(3):
                                    dereference_monitors.append(DereferencesMonitor(HFTokenizerWrapper(tokenizer), filebuffer))
                                # dereference_monitor = DereferencesMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                                mgd_logits_processor_dereference = MGDLogitsProcessor(dereference_monitors, lsp.language_server.server.loop)
                                input_ids = tokenizer(query['augmented_prompt'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                                logits_processor = LogitsProcessorList([mgd_logits_processor_dereference]) 
                                sucesss_d_p = generate_with_post_processing(model, 
                                                                            tokenizer, 
                                                                            input_ids, 
                                                                            query, 
                                                                            stopping_criteria, 
                                                                            language, 
                                                                            oom_task_id, 
                                                                            logits_processor)
                            if not using_postprocessing or not sucesss_d_p:
                                dereference_monitor = DereferencesMonitor(HFTokenizerWrapper(tokenizer), filebuffer)
                                mgd_logits_processor_dereference = MGDLogitsProcessor([dereference_monitor], lsp.language_server.server.loop)
                                input_ids = tokenizer(query['augmented_prompt'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                                logits_processor = LogitsProcessorList([mgd_logits_processor_dereference])
                                generated_ids = model.generate(
                                    input_ids,
                                    do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=50,
                                    stopping_criteria=stopping_criteria,
                                    logits_processor=logits_processor,
                                    # early_stopping=True,
                                )
                                predict = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens = True).split('\n')[0]
                                query['predict'] = predict
                            if not sucesss_d_p:
                                print('fail')
                                break
                            
                            
                except Exception as e:
                    # print(1)
                    # print(e)
                    traceback.print_exc()
                    # torch.cuda.empty_cache()
                    # print('Fail to start language server..')
                    time.sleep(10)
            if 'predict' not in query:
                continue
                    
            res = {
                    "prompt": query["augmented_prompt"],
                    "metadata": query["metadata"],
                    "groundtruth": query["groundtruth"],
                    "predict": query["predict"]
                }
            
            if "output" in query:
                res["output"] = query["output"]

            if "error" in query:
                res["error"] = query["error"]

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(res))
                f.write("\n")

            if i%20 == 0:
                evaluate(output_file)
                


    else:
        with torch.no_grad():
            for i, query in enumerate(tqdm(data), 1):

                if query["metadata"]["task_id"] in has_predicted:
                    continue
         

                input_ids = tokenizer(query['augmented_prompt'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                
                if using_postprocessing:
                    if not generate_with_post_processing(model, tokenizer, input_ids, query, stopping_criteria, language, oom_task_id):
                        continue

                if not (using_postprocessing or using_decoding):
                    generated_ids = model.generate(
                        input_ids,
                        pad_token_id = tokenizer.eos_token_id,
                        max_new_tokens = 64,
                        stopping_criteria = stopping_criteria,
                    )
                    query['predict'] = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens = True).strip()
                # except Exception as e:
                #     print(query["metadata"]["task_id"])
                #     print(e)
                #     continue
                # if generated_ids == None:
                #     continue

                if not 'predict' in query:
                    query['predict'] = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens = True).strip()

                res = {
                        "prompt": query["augmented_prompt"],
                        "metadata": query["metadata"],
                        "groundtruth": query["groundtruth"],
                        "predict": query["predict"]
                    }
                
                if "output" in query:
                    res["output"] = query["output"]

                if "error" in query:
                    res["error"] = query["error"]

                if not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "a") as f:
                    f.write(json.dumps(res))
                    f.write("\n")

                if i%20 == 0:
                    evaluate(output_file)
    
    evaluate(output_file)
    print("MODEL:", args.model)
    print(f'OOM tasks: {len(oom_task_id)}')
    print("TYPE:", args.type)
    print("OUTPUT_FILE:", output_file)

    sava_oom_errs(os.path.join(os.path.dirname(output_file), 'oom_err'), os.path.basename(output_file), oom_task_id)
    # dump_jsonl(data, output_file)
    

if __name__ == "__main__":
    args = get_args()
    using_postprocessing = ('p' in args.type)

    if args.language == 'java':
        if not using_postprocessing:
            output_file = f"data/generate/study/{args.language}/{args.model}-{args.type}.jsonl"
        else:
            output_file = f"data/generate/study/{args.language}/deepseek-coder-p/{args.model}-{args.type}.jsonl"
    else:
        output_file = f"data/generate/study/{args.model}-{args.type}.jsonl"

    # load model
    print("MODEL:", args.model)
    print("TYPE:", args.type)
    print("OUTPUT_FILE:", output_file)
    print("LANGUAGE:", args.language)
    print(f"load model to cuda: {args.gpu}")
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    model, tokenizer = init_model(args.model, device)#, using_postprocessing)
    try:
        MODEL_VECTOR_SIZE = MODEL_VECTOR_SIZE[args.model]
    except:
        MODEL_VECTOR_SIZE = tokenizer.vocab_size

    # load dataset
    data = load_data(args)

    using_decoding = ('d' in args.type)
    
    
    predict(model, tokenizer, data, output_file, using_decoding, using_postprocessing)