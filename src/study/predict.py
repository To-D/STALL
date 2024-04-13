
# -*- coding: utf-8 -*-

import argparse
import os
import json
from typing import List
from pygtrie import CharTrie

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, \
      StoppingCriteriaList, LogitsProcessorList, LogitsProcessor

from utils import load_jsonl, dump_jsonl, REPOS_DIR, MODEL_VECTOR_SIZE
from prompt_builder import PromptBuilder
from evaluate import evaluate
from analysize import analysize
from feature import is_API_invocation_or_completion
from postprocessing import check_syntax

from parso.cache import clear_cache, _get_default_cache_path

def clear_sa_cache():
    cache_path = _get_default_cache_path()
    if not os.path.exists(cache_path):
        try:
            os.mkdir(cache_path)
            clear_cache()
        except:
            return

def init_model(model_id, device):
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
        if token.startswith("\n"):
            newline_token.append(i)
    for sidx in symbols:
        symbol_probability[sidx] = 1
    return symbol_probability, newline_token

def static_analysis(query, symbol_embeddings):
    options = analysize(REPOS_DIR["python"], query)
    clear_sa_cache()
    if options:
        analysis_probobility = torch.zeros(MODEL_VECTOR_SIZE)
        for option in options:
            if option['complete'] == '' and query['prompt'].endswith(option['token']):
                for idx, i in enumerate(symbol_embeddings):
                    analysis_probobility[idx] = i
                prefix = option['token']
                for option in options:
                    if option['complete'] and option['token'].startswith(prefix):
                        next_token_id = tokenizer(option['complete'], add_special_tokens=False).input_ids[0]
                        analysis_probobility[next_token_id] = 1
                break
            # 有大小写相同的情况，如AAA和aaa，都会返回''
            if option['complete'] == '': continue
            next_token_ids = tokenizer(option['complete'], add_special_tokens=False).input_ids
            if next_token_ids:
                next_token_id = next_token_ids[0]
                analysis_probobility[next_token_id] = 1
        return analysis_probobility, options
    else:
        return None, []

class SALogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        query = self.query
        query["output"][self.turn] = {}
        if len(input_ids[0]) > query["input_len"]:
            next_token = tokenizer.decode(input_ids[0][-1], skip_special_tokens= True)
            query["prompt"] += next_token
            query['metadata']["groundtruth_start_colno"] += len(next_token)
            query["output"][self.turn-1]["predict"] = next_token

        self.turn += 1

        if not is_API_invocation_or_completion(query["prompt"]):
            return scores

        sa_prob, options = static_analysis(query, self.symbol_embeddings)
        query["output"][self.turn-1]["options"] = options

        if len(options)==0:
            return scores
        scores.to(device)
        sa_prob.to(device)
        scores[0][sa_prob == 0] = float('-inf')

        # ## 提取前5名的token
        # # 用softmax函数转换为概率分布
        # probabilities = torch.softmax(scores, dim=-1)

        # # 提取top-5概率及其索引
        # top5_probabilities, top5_indices = torch.topk(probabilities, 5)

        # top_probs_list = top5_probabilities.detach().cpu().numpy().tolist()
        # top_idxs_list = top5_indices.cpu().numpy().tolist()

        # print("Top 5 probabilities:", top_probs_list)
        # print("Top 5 indices:", top_idxs_list)
        # for idx, id in enumerate(top_idxs_list[0]):
        #     token  = tokenizer.decode(id, skip_special_tokens = True)
        #     print(token, top5_probabilities[0][idx].item())
        # input()

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
        if self.stop_conditions.int().sum().item() == input_ids.shape[0]:
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
    """
    parser.add_argument('-t', '--type',
                        required=True,
                        choices=["i","r", "si", "sls", "sd", "sp", "si-d", "si-p", "sls-d", "sls-p",
                                  "r-si", "r-sls", "r-sd", "r-sp", "r-si-d", "r-si-p", "r-sls-d", "r-sls-p"],
                        help=type_info)
    parser.add_argument('-i', '--input', type=str, help="input file")
    parser.add_argument('-g', '--gpu', type=int, help="gpu id")
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


def predict(model, tokenizer, data, output_file, using_decoding=False, using_postprocessing=False, language='python'):
    has_predicted = []
    if os.path.exists(output_file):
        for line in load_jsonl(output_file):
            has_predicted.append(line["metadata"]["task_id"])

    stopping_criteria = StoppingCriteriaList()
    symbol_embeddings, newline_token_list = get_symbol_embeddings()
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=newline_token_list))
    if using_decoding:
        logits_processor = SALogitsProcessor()

    with torch.no_grad():
        for i, query in enumerate(tqdm(data), 1):

            if query["metadata"]["task_id"] in has_predicted:
                continue
            try:
                input_ids = tokenizer(query['augmented_prompt'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)

                if using_decoding:
                    query["input_len"] = len(input_ids[0])
                    query["output"] = {}
                    logits_processor.set_turn(1)
                    logits_processor.set_query(query)
                    logits_processor.set_symbol_embeddings(symbol_embeddings)
                    generated_ids = model.generate(
                        input_ids,
                        pad_token_id = tokenizer.eos_token_id,
                        max_new_tokens = 64,
                        logits_processor=LogitsProcessorList([logits_processor]),
                        stopping_criteria = stopping_criteria,
                    )
                    output = ""
                    for _, v in query["output"].items():
                        if "predict" in v:
                            output += v["predict"]
                    query["predict"] = output


                if using_postprocessing:
                    try:
                        generated_ids = model.generate(
                            input_ids,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 64,
                            stopping_criteria = stopping_criteria,
                            num_beams=3,
                            num_return_sequences=3
                        )
                        has_valid_output = False
                        outputs = []
                        for d in generated_ids:
                            outputs.append(tokenizer.decode(d[len(input_ids[0]):], skip_special_tokens=True).strip().split("\n")[0])

                        query["output"] = outputs

                        query["error"] = {}
                        for idx, output in enumerate(outputs, 1):
                            code = query["metadata"]["prefix_src"] + query["prompt"] + output
                            filepath = REPOS_DIR[language] + "/"  + query["metadata"]["repository"] + "/" + query["metadata"]["file"]
                            without_error, error_message = check_syntax(code, filepath, query["metadata"]["repository"], query["metadata"]["groundtruth_start_lineno"]+1)
                            if without_error:
                                has_valid_output = True
                                query["predict"] = output
                                break
                            else:
                                query["error"][idx] = {"predict":output ,"err_message":error_message}
                        if not has_valid_output:
                            query["predict"] = outputs[0]
                    except:
                        print("repairing")
                        generated_ids = model.generate(
                            input_ids,
                            pad_token_id = tokenizer.eos_token_id,
                            max_new_tokens = 64,
                            stopping_criteria = stopping_criteria,
                        )

                if not (using_postprocessing or using_decoding):
                    generated_ids = model.generate(
                        input_ids,
                        pad_token_id = tokenizer.eos_token_id,
                        max_new_tokens = 64,
                        stopping_criteria = stopping_criteria,
                    )
            except Exception as e:
                print(query["metadata"]["task_id"])
                print(e)
                continue

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

            with open(output_file, "a") as f:
                f.write(json.dumps(res))
                f.write("\n")

            if i%20 == 0:
                evaluate(output_file)
    evaluate(output_file)
    print("MODEL:", args.model)
    print("TYPE:", args.type)
    print("OUTPUT_FILE:", output_file)
    # dump_jsonl(data, output_file)

if __name__ == "__main__":
    args = get_args()

    output_file = f"data/generate/study/{args.language}/{args.model}-{args.type}.jsonl"

    # load model
    print("MODEL:", args.model)
    print("TYPE:", args.type)
    print("OUTPUT_FILE:", output_file)
    print("LANGUAGE:", args.language)
    print(f"load model to cuda:{args.gpu}")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model, tokenizer = init_model(args.model, device)
    try:
        MODEL_VECTOR_SIZE = MODEL_VECTOR_SIZE[args.model]
    except:
        MODEL_VECTOR_SIZE = tokenizer.vocab_size

    # load dataset
    data = load_data(args)

    using_decoding = ('d' in args.type)
    using_postprocessing = ('p' in args.type)
    predict(model, tokenizer, data, output_file, using_decoding, using_postprocessing)