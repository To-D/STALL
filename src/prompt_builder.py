import os
from tqdm import tqdm
import asyncio
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from retrieve.Jaccard import retrieve_jaccard_single
from analysize import analysize, get_eclipse_ls_complete
from parso.cache import clear_cache, _get_default_cache_path
from import_analysizer import import_analysize
from eclipse_jdt_ls import multi_request_get_completion
from utils import load_jsonl, dump_jsonl, REPOS_DIR

def clear_sa_cache():
    cache_path = _get_default_cache_path()
    if not os.path.exists(cache_path):
        try:
            os.mkdir(cache_path)
        except:
            clear_cache()
    clear_cache()

def transform_format(data):
    for d in tqdm(data):
        metadata = d["metadata"]
        metadata["ground_truth"] = d["groundtruth"]
        metadata["fpath_tuple"] = (metadata["repository"] + "/" + metadata["file"]).split("/")
        metadata["context_start_lineno"] = metadata["context_start_lineno"]
        metadata["line_no"] = metadata["groundtruth_start_lineno"]


class ContextFactory:
    @staticmethod
    def construct_infile_context(case, tokenizer, max_length=2000):
        input_id = tokenizer(case["metadata"]["prefix_src"] + case["prompt"], \
                                return_tensors="pt").input_ids[0]
        # length of infile context <= 2000
        if len(input_id) > max_length:
            input_id = input_id[-max_length:]
        return tokenizer.decode(input_id, skip_special_tokens=True)

    @staticmethod
    def construct_retrieve_context(case, tokenizer, language='python', max_length=3000):
        repo_dir = REPOS_DIR[language]
        case = retrieve_jaccard_single(case, repo_dir, 20, 2, language)[0]
        top = "# Here are some similar code fragments in other files of this repository\n"
        retrieval_context_len = len(tokenizer(top,return_tensors="pt").input_ids[0])
        retrieval_context = ""
        for code in case["similar_code"]:
            partial_context = f"# {'/'.join(code['metadata'][0]['fpath_tuple'])}\n"
            partial_context += '\n'.join('# ' + line for line in code['next_k_window'].split('\n')) + '\n'
            partial_context += "\n"
            partial_context_len = len(tokenizer(partial_context,return_tensors="pt").input_ids[0])
            if retrieval_context_len + partial_context_len > max_length:
                break
            else:
                retrieval_context = partial_context + retrieval_context
                retrieval_context_len += partial_context_len
        return top+retrieval_context

    def construct_language_server_context(case, tokenizer, max_length=3000) -> str:
        repo_dir = REPOS_DIR[language]
        clear_sa_cache()
        options = analysize(repo_dir, case)
        black_list = ["numpy", "torch", "transformer", "pandas", "datasets"]
        if case["prompt"].endswith("torch.") or case["prompt"].endswith("np.") or case["prompt"].endswith("numpy."):
            return ""
        lines = []
        stmts = []
        methods = []
        for option in options:
            is_third_party = False
            for third_patry in black_list:
                if option["fullname"] and option["fullname"].startswith(third_patry):
                    is_third_party = True
                    break
            if is_third_party:
                continue
            token = option["token"]
            if not token.startswith("__"):
                if option["type"] == "method":
                    methods.append(option["signature"])
                else:
                    stmts.append(token)

        lines = []
        if methods or stmts:
            lines.append("## These APIs can be invoked")
            for s in stmts:
                lines.append(f"# {s}")
            for m in methods:
                lines.append(f"# {m}")

        context = "\n".join(lines)

        input_id = tokenizer(context, return_tensors="pt").input_ids[0]
        if len(input_id) > max_length:
            input_id = input_id[-max_length:]
        context = tokenizer.decode(input_id, skip_special_tokens=True)

        return context

    def construct_eclipse_jdt_ls_context(datas, tokenizer, max_length=3000):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        repo_dict = {}
        for data in datas:
            repo = data['metadata']['repository']
            if repo in repo_dict.keys():
                repo_dict[repo].append(data)
            else:
                repo_dict[repo] = [data]
        for key, value in repo_dict.items():
            print(f'processing repo: {key}..')
            asyncio.run(multi_request_get_completion(key, value))

    def construct_import_context(case, tokenizer, language='python', max_length=3000) -> str:
        sources = import_analysize(case, language)
        context_lines = []
        context_lines.append("# Here are some definitions in related files of this repository")
        for key, value in sources.items():
            context_lines.append(f"\n# {case['metadata']['repository'] + '/' + value['path']}")
            for line in value["content"].split("\n"):
                context_lines.append("# " + line)

        context = "\n".join(context_lines)
        input_id = tokenizer(context, return_tensors="pt").input_ids[0]
        if len(input_id) > max_length:
            input_id = input_id[-max_length:]
        context = tokenizer.decode(input_id, skip_special_tokens=True)

        return context

class PromptBuildStrategy(ABC):
    @abstractmethod
    def build(self, case):
        pass

class InfileContextBuilder(PromptBuildStrategy):
    def build(self, cases):
        for case in cases:
            case["augmented_prompt"] = case["infile_context"]

class RAGContextBuilder(PromptBuildStrategy):
    def build(self, cases) -> str:
        # # same tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("data//models/deepseek-coder-6.7b-base")
        # transform_format(cases)
        # for case in tqdm(cases):
        #     infile_context = ContextFactory.construct_infile_context(case, tokenizer)
        #     retrieve_context = ContextFactory.construct_retrieve_context(case, tokenizer)
        #     case["augmented_prompt"] = retrieve_context + f"# current file: {'/'.join(case['metadata']['fpath_tuple'])}\n" + infile_context
        for case in cases:
            case["augmented_prompt"] = case["retrieve_context"] + \
                f"# current file: {'/'.join(case['metadata']['fpath_tuple'])}\n" + case["infile_context"]

class LanguageServerPromptBuilder():
    def build(self, cases) -> str:
        # tokenizer = AutoTokenizer.from_pretrained("data//models/deepseek-coder-6.7b-base")
        # for case in tqdm(cases):
        #     infile_context = ContextFactory.construct_infile_context(case, tokenizer)
        #     static_analysis_context = ContextFactory.construct_language_server_context(case, tokenizer)
        #     lines = infile_context.split("\n")
        #     case["augmented_prompt"] = "\n".join(lines[:-1]) + "\n" + static_analysis_context + "\n" + lines[-1]
        for case in cases:
            prompt = ""
            lines = case["infile_context"].split("\n")
            prompt += "\n".join(lines[:-1])
            prompt += f"\n{case['ls_context']}"
            prompt += "\n" + lines[-1]
            case["augmented_prompt"] = prompt

class ImportContextBuilder():
    def build(self, cases) -> str:
        # tokenizer = AutoTokenizer.from_pretrained("data//models/deepseek-coder-6.7b-base")
        # for case in tqdm(cases):
        #     infile_context = ContextFactory.construct_infile_context(case, tokenizer)
        #     static_analysis_context = ContextFactory.construct_import_context(case, tokenizer)
        #     case["augmented_prompt"] = static_analysis_context + f"\n\n# current file: {case['metadata']['repository']}/{case['metadata']['file']}\n" + infile_context
        for case in cases:
            case["augmented_prompt"] = case["import_context"] + \
                f"\n\n# current file: {case['metadata']['repository']}/{case['metadata']['file']}\n" + case["infile_context"]

class RG_ImportContextBuilder():
    def build(self, cases) -> str:
        for case in cases:
            case["augmented_prompt"] = ""
            if case["import_context"]:
                case["augmented_prompt"] += case["import_context"] + "\n\n"
            case["augmented_prompt"] += "#" + case["retrieve_context"] + \
            f"\n# current file: {case['metadata']['repository']}/{case['metadata']['file']}\n" + case["infile_context"]

class RG_LSContextBuilder():
    def build(self, cases) -> str:
        for case in cases:
            lines = case["infile_context"].split("\n")
            prompt = case["retrieve_context"]
            prompt += "\n".join(lines[:-1])
            prompt += f"\n{case['ls_context']}"
            prompt += "\n" + lines[-1]
            case["augmented_prompt"] = prompt

class PromptBuilder:
    def __init__(self, prompt_type):
        self.builder = None
        if prompt_type == 'i':
            self.builder = InfileContextBuilder()
        elif prompt_type == 'r':
            self.builder = RAGContextBuilder()
        elif prompt_type == 'si':
            self.builder = ImportContextBuilder()
        elif prompt_type == 'sls':
            self.builder = LanguageServerPromptBuilder()
        elif prompt_type == 'sd':
            self.builder = InfileContextBuilder()
        elif prompt_type == 'sp':
            self.builder = InfileContextBuilder()
        elif prompt_type == 'r-si':
            self.builder = RG_ImportContextBuilder()
        elif prompt_type == 'r-si-d':
            self.builder = RG_ImportContextBuilder()
        elif prompt_type == 'r-si-p':
            self.builder = RG_ImportContextBuilder()
        elif prompt_type == 'r-sd':
            self.builder = RAGContextBuilder()
        elif prompt_type == 'r-sp':
            self.builder = RAGContextBuilder()
        elif prompt_type == 'r-sls':
            self.builder = RG_LSContextBuilder()
        elif prompt_type == 'r-sls-d':
            self.builder = RG_LSContextBuilder()
        elif prompt_type == 'r-sls-p':
            self.builder = RG_LSContextBuilder()
        elif prompt_type == 'si-d':
            self.builder = ImportContextBuilder()
        elif prompt_type == 'si-p':
            self.builder = ImportContextBuilder()
        elif prompt_type == 'sls-d':
            self.builder = LanguageServerPromptBuilder()
        elif prompt_type == 'sls-p':
            self.builder = LanguageServerPromptBuilder()
        elif prompt_type == 'si-d-p':
            self.builder = ImportContextBuilder()
        elif prompt_type == 'r-si-d-p':
            self.builder = RG_ImportContextBuilder()

    def build(self, cases):
        return self.builder.build(cases)


if __name__ == "__main__":
    # data = load_jsonl("data/datasets/cceval/cceval_2460.jsonl")
    # output_path = "data/datasets/cceval/cceval_2460_with_augmented_context.jsonl"

    language = 'java'
    data = load_jsonl("data/datasets/cceval/java/cceval_java.jsonl")
    output_path = "data/datasets/cceval/java/cceval_java_with_augmented_context.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("data/models/deepseek-coder-6.7b-base")
    # infile context
    print("Constructing Infile context...")
    for case in data:
        infile_context = ContextFactory.construct_infile_context(case, tokenizer)
        case["infile_context"] = infile_context

    print("Constructing retrieve context...")
    transform_format(data)
    # retrieve context
    for case in tqdm(data):
        retrieve_context = ContextFactory.construct_retrieve_context(case, tokenizer, language)
        case["retrieve_context"] = retrieve_context

    # language server context
    print("Constructing language server context...")
    if language == 'python':
        for case in tqdm(data):
            language_server_context = ContextFactory.construct_language_server_context(case, tokenizer)
            case["ls_context"] = language_server_context
    elif language == 'java':
        ContextFactory.construct_eclipse_jdt_ls_context(data, tokenizer)

    # import context
    print("Constructing import context...")
    for case in tqdm(data):
        import_context = ContextFactory.construct_import_context(case, tokenizer, language)
        case["import_context"] = import_context

    dump_jsonl(data, output_path)