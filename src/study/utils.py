import json
import jsonlines
import pickle
import os
import re
from functools import lru_cache
from nltk.tokenize import RegexpTokenizer
import keyword
from typing import FrozenSet

MODEL_VECTOR_SIZE = {
    "deepseek-coder": 32256,
    "starcoder": 49152,
    "codellama": 32016,
}
REPOS_DIR = {
    "python": "data/repos/cceval/python",
    "java": "data/repos/cceval/java"
}
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")
code_tokenizer = RegexpTokenizer(r'\w+')
__all__ = ['get_language_keywords']

_LANGUAGE_TO_FILENAME = {
    'c': 'c.txt',
    'cpp': 'cpp.txt',
    'c++': 'cpp.txt',
    'csharp': 'csharp.txt',
    'c_sharp': 'csharp.txt',
    'c#': 'csharp.txt',
    'go': 'go.txt',
    'java': 'java.txt',
    'javascript': 'javascript.txt',
    'js': 'javascript.txt',
    'php': 'php.txt',
    'ruby': 'ruby.txt',
    'typescript': 'typescript.txt',
    'ts': 'typescript.txt',
}

@lru_cache()
def get_language_keywords(language: str) -> FrozenSet[str]:
    """
    Returns the keywords of a programming language.

    There are some inconsistencies across languages wrt to
    what is considered a keyword. For example, the true/false
    literals are considered keywords in many languages. However,
    we exclude them here for consistency. We also exclude special
    functions-like keywords, such as `die()` in PHP.
    """
    language = language.lower()
    if language == 'python':
        return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')
    elif language in _LANGUAGE_TO_FILENAME:
        name = _LANGUAGE_TO_FILENAME[language]
        with open(os.path.join(os.path.dirname(__file__), name)) as f:
            return frozenset(l.strip() for l in f if len(l.strip()) > 0)
    else:
        raise Exception('Language keywords `%s` not supported yet. Consider contributing it to dpu-utils.' % language)

def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def load_jsonl_2(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def load_jsonl(file):
    data = []
    with jsonlines.open(file, "r") as f:
        for line in f:
            data.append(line)
    return data

def dump_jsonl(data, output_file):
    with jsonlines.open(output_file,"w") as f:
        for line in data:
            f.write(line)

def get_last_token(prompt):
    with open("python-seperator.json", "r") as f:
        seperators = json.load(f)
    i = len(prompt)-1
    while i>=0:
        if prompt[i] in seperators:
            return i+1, prompt[i+1:]
        i-=1

def collect_symbol_tokens(tokenizer):
    import json
    idxs = []
    with open("python-symbol.json", "r") as f:
        symbols = json.load(f)
    for idx in range(50295):
        token = tokenizer.decode([idx])
        if token[0] in symbols:
            idxs.append(idx)
    with open("symbol_idxs.json", "w+") as f:
        json.dump(idxs, f)

def is_identifier(token, lang=None):
    return True if IDENTIFIER_REGEX.match(token) \
                   and (lang is None or token not in get_language_keywords(lang)) \
        else False

def extract_identifiers(source_code, lang = 'python'):
    # the main idea is to remove String from a source code
    # then, tokenize the code to get all words and match with identifier regular expression
    # check if it is a language specific keyword, it not, then it is an identifier
    source_code_without_strings = re.sub(string_pattern, '', source_code)
    _ids = [t for t in code_tokenizer.tokenize(source_code_without_strings) if is_identifier(t, lang)]
    return _ids

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenTokenizerFast
    tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-6B-mono")
    collect_symbol_tokens(tokenizer)