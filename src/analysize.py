import jedi
import os
import keyword
import sys
import json
from tqdm import tqdm
import traceback

def add_sys_path(file_path, repo_name, repo_dir):
    added_path = []
    # 文件同级目录
    filepath = file_path
    added_path.append(os.path.dirname(filepath))

    # 仓库目录，和其下的src目录，和其下的同名子目录
    sub_repo_dir = repo_dir + "/" + repo_name
    added_path.append(sub_repo_dir)
    added_path.append(sub_repo_dir + "/src")
    added_path.append(sub_repo_dir + "/" + repo_name)

    # 上一级目录
    added_path.append(os.path.dirname(os.path.dirname(filepath)))

    # 如果含src文件夹，则将测试文件夹的父目录和它本身加入
    src_index = filepath.find("src")
    if src_index != -1:
        added_path.append(filepath[:src_index])
        added_path.append(filepath[:src_index+3])

    # 如果含测试文件夹，则将测试文件夹的父目录加入
    test_index = filepath.find("tests")
    if test_index != -1:
        added_path.append(filepath[:test_index])

    # 如果含app文件夹，则将app文件夹的父目录加入
    app_index = filepath.find("app")
    if app_index != -1:
        added_path.append(filepath[:app_index])
        added_path.append(filepath[:app_index+3])

    # 上上级目录
    added_path.append(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))
    added_path = [path for path in added_path if os.path.exists(path)]
    return added_path

class StaticAnalysis():
    def __init__(self, repo_dir) -> None:
        """
        :param repo_dir: 存放各仓库的文件夹的绝对路径
        """
        self.repo_dir = repo_dir
        self.keywords = keyword.kwlist

        self.standard_library_modules = [module for module in sys.modules
                                         if module.startswith('builtins.') or module.startswith('sys.')]



    def get_complete(self, code, repo_name, fpath, line, column) -> list:
        """
        :param code: 所有前文
        :param repo_name: 仓库名
        :param fpath: 待补全文件的路径
        :param line: 从1开始计算的行号
        :param column: 从0开始计算的列号

        :return resultsL: lsit[str] 候选的补全内容
        """


        # 行首不进行静态分析补全
        if column == 0:
            return []


        path = os.path.join(self.repo_dir, repo_name)
        project = jedi.Project(path = path)
        added_path = add_sys_path(fpath, repo_name, self.repo_dir)
        project.added_sys_path.extend(added_path)

        is_self = (code[-5:] == 'self.')

        script = jedi.Script(code = code, path = fpath, project = project)

        completes = script.complete(line + 1, column)

        results = []
        for complete in completes:
            name = complete.name
            suffix = complete.complete

            # 大小写问题
            if name != suffix and not code.endswith(name[:-len(suffix)]):
                continue

            # complete.type 容易出问题 若是报错则直接跳过
            try:
                complete_dict = {
                    "type": complete.type,
                    "complete": suffix,
                    "token": name,
                    "signature": complete.docstring().split('\n')[0],
                    "docstring": '\n'.join(complete.docstring().split('\n')[1:])
                }
            except Exception as e:
                print("[INNER ERROR]", e)
                complete_dict = {
                    "type": "statement",
                    "complete": suffix,
                    "token": name,
                    "signature": "",
                    "docstring": ""
                }
                continue

            if complete_dict['type'] in ['function', 'class']:
                complete_dict['type'] = 'method'
            else:
                complete_dict['docstring'] = complete.docstring()
                complete_dict['signature'] = ''

            complete_dict['fullname'] = complete.full_name
            results.append(complete_dict)
        return results

def analysize(repo_dir, sample):
    src = sample['metadata']['prefix_src'] + sample['prompt']
    if 'repository' in sample['metadata']:
        repo_name = sample['metadata']['repository']
        fpath = os.path.join(repo_dir, repo_name, sample['metadata']['file'])
        line_no = sample['metadata']['groundtruth_start_lineno']
        column_no = sample['metadata']['groundtruth_start_colno']
    else:
        repo_name = sample['metadata']['fpath_tuple'][0]
        fpath = os.path.join(repo_dir, "/".join(sample["metadata"]['fpath_tuple']))
        line_no = sample['metadata']['line_no']
        column_no = sample['metadata']['groundtruth_start_colno']

    jedi_analysis = StaticAnalysis(repo_dir)
    try:
        results = jedi_analysis.get_complete(src, repo_name, fpath, line_no, column_no)
    except Exception as e:
        results = []
        print("Error=>", e)
    return results

def analysize_batches(repo_dir, queries_data):
    for sample in queries_data:
        src = sample['metadata']['prefix_src'] + sample['prompt']
        repo_name = sample['metadata']['repository']
        fpath = os.path.join(repo_dir, repo_name, sample['metadata']['file'])
        line_no = sample['metadata']['groundtruth_start_lineno']
        column_no = sample['metadata']['groundtruth_start_colno']

        jedi_analysis = StaticAnalysis(repo_dir)
        try:
            results = jedi_analysis.get_complete(src, repo_name, fpath, line_no, column_no)
        except Exception as e:
            results = []

        sample['analysis_options'] = results
    return queries_data