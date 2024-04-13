"""
    使用eclipse.jdt.ls的python实现multilspy来对cceval的java数据集静态分析
"""

import asyncio
import os
import shutil
import re
import tqdm
from enum import Enum
import time
import sys

sys.path.append('PriorWorks/monitors4codegen-lsp/src')
sys.path.append('PriorWorks/monitors4codegen-lsp')
from monitors4codegen.multilspy import SyncLanguageServer, LanguageServer
from monitors4codegen.multilspy import multilspy_settings
from monitors4codegen.multilspy.multilspy_config import MultilspyConfig
from monitors4codegen.multilspy.multilspy_logger import MultilspyLogger
from utils import load_jsonl, dump_jsonl, REPOS_DIR

LANGUAGE = 'java'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class REQUEST_TYPE(Enum):
    COMPLETION = 1
    DEFINITION = 2

def clear_folder_contents(folder_path):
    if not os.path.isdir(folder_path):
        print(f"unvalid folder {folder_path}")
        return
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        except Exception as e:
            print(f"an error occurs when deleting {item_path}: {e}")


async def single_request(repo: str, fpath: str, line: int, col: int, req_type: REQUEST_TYPE):
    """Request for single data
    repo: root path
    fpath: relative path to java file
    line: line number of symbol for which request is being made 0-based
    col: column number of symbol for which request is being made 0-based
    req_type: request type 

    """
    config = MultilspyConfig.from_dict({"code_language": LANGUAGE}) # Also supports "python", "rust", "csharp"
    logger = MultilspyLogger()
    lsp = LanguageServer.create(config, logger, repo)
    async with lsp.start_server():

        if req_type == REQUEST_TYPE.COMPLETION:
            results = await lsp.request_completions(
                fpath, 
                line,
                col,
                allow_incomplete=False
            )
        elif req_type == REQUEST_TYPE.DEFINITION:
            results = await lsp.request_definition(
                fpath, 
                line,
                col 
            )
        return results

async def multi_request_get_import(repo: str, datas: list):
    """处理import
    """
    config = MultilspyConfig.from_dict({"code_language": LANGUAGE})
    logger = MultilspyLogger()
    repo_dir = os.path.join(REPOS_DIR[LANGUAGE], repo)
    lsp = LanguageServer.create(config, logger, repo_dir)
    async with lsp.start_server():
        for data in tqdm.tqdm(datas):
            data['import_definition'] = {}
            abs_fpath = os.path.join(repo_dir, data['metadata']['file'])
            with open(abs_fpath, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('import '):
                    col = line.find(';') - 1
                    assert col > 0
                    rets = await lsp.request_definition(data['metadata']['file'], i, col)

                    # 若有返回内容，并且是本仓库的import
                    # jar uri -> "jdt://xxx"
                    # repo uri -> "file:///xxx"
                    if len(rets) > 0 and rets[0]['uri'].startswith('file'):
                        # 确认import的definition返回的结果只有一条
                        assert len(rets) == 1
                        data['import_definition'][line.strip()] = rets

async def multi_request_get_completion(repo: str, datas: list):
    config = MultilspyConfig.from_dict({"code_language": LANGUAGE})
    logger = MultilspyLogger()
    repo_dir = os.path.join(REPOS_DIR[LANGUAGE], repo)
    lsp = LanguageServer.create(config, logger, repo_dir)
    # cache_dir = multilspy_settings.MultilspySettings.get_global_cache_directory()
    # clear_folder_contents(cache_dir)
    async with lsp.start_server():
        for data in tqdm.tqdm(datas, leave=False):
            # abs_fpath = os.path.join(repo_dir, data['metadata']['file'])
            rets = await lsp.request_completions(data['metadata']['file'], 
                                                data['metadata']['groundtruth_start_lineno'], 
                                                data['metadata']['groundtruth_start_colno'])
            
            methods = []
            stmts = []
            for ret in rets: 
                if ret['kind'] == 2: # lsp 协议说明kind为2则是method
                    methods.append(ret['detail'])
                else:
                    try:
                        stmts.append(ret['detail'])
                    except:
                        stmts.append(ret['completionText'])

            lines = []
            if methods or stmts:
                lines.append("## These APIs can be invoked")
                for s in stmts:
                    lines.append(f"# {s}")
                for m in methods:
                    lines.append(f"# {m}")

            context = "\n".join(lines)
            data["ls_context"] = context

def get_import_src_for_impt_stmt(abs_fpath) -> dict:
    """
    根据传入的java文件绝对路径，返回import内容。
    """
    import_definitions = {}
    with open(abs_fpath, 'r') as f:
        lines = f.readlines()
    
    # 获取到包名
    package = None
    for line in lines:
        if line.lstrip().startswith('package'):
            pattern = r"package\s+(.*?);"
            matches = re.findall(pattern, line)
            package = matches[0]
            break
    
    if package == None:
        return import_definitions
    
    for line in lines:
        if line.startswith('import '):
            pattern = r"import\s+(.*?);"
            matches = re.findall(pattern, line)
            import_pkg = matches[0]
            pkg_i = abs_fpath.find(package.replace('.', '/'))
            import_uri = os.path.join(abs_fpath[:pkg_i], import_pkg.replace('.', '/') + '.java')

            # 是仓库内导入
            if os.path.exists(import_uri):
                import_definitions[line.strip()] = import_uri
    return import_definitions

def get_import():
    """使用正则处理import
    """
    datas = load_jsonl('data/datasets/cceval/java/cceval_java.jsonl')
    start = time.time()

    for data in tqdm.tqdm(datas):
        import_definitions = get_import_src_for_impt_stmt(os.path.join(REPOS_DIR['java'], data['metadata']['repository'], data['metadata']['file']))
        data['import_definition'] = import_definitions

    dump_jsonl(datas, 'data/datasets/cceval/java/cceval_java_import_info_2.jsonl')
    print(f'total costs {time.time() - start}s')
                
def lsp():
    datas = load_jsonl('data/datasets/cceval/java/cceval_java.jsonl')

    start = time.time()

    # 将数据根据仓库归类
    repo_dict = {}
    for data in datas:
        repo = data['metadata']['repository']
        if repo in repo_dict.keys():
            repo_dict[repo].append(data)
        else:
            repo_dict[repo] = [data]

    # 逐仓库进行import处理
    for key, value in repo_dict.items():
        print(f'processing repo: {key}..')
        asyncio.run(multi_request_get_import(key, value))
        dump_jsonl(datas, 'data/datasets/cceval/java/cceval_java_import_info.jsonl')
    
    print(f'total costs {time.time() - start}s')


if __name__ == '__main__':
    # get_import()
    datas = load_jsonl('data/datasets/cceval/java/cceval_java.jsonl')

    
    # data = datas[21]
    for d in datas:
        if d['metadata']['task_id'] == 'project_cc_java/844':
            data = d
            break
    
    repo = data['metadata']['repository']
    fpath = data['metadata']['file']
    line = data['metadata']['groundtruth_start_lineno']
    
    col = 68 # data['metadata']['groundtruth_start_colno']
    abs_path = os.path.join(REPOS_DIR['java'], repo, fpath)
    with open(abs_path, 'r') as f:
        lines = f.readlines()
        print(f'target line: {lines[line][:col]}')
    rets = asyncio.run(single_request(os.path.join(REPOS_DIR['java'], repo), fpath, line, col, REQUEST_TYPE.COMPLETION))
    for ret in rets:
        print(ret)





