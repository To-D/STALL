import ast
import os
import re
import sys

import jedi
from utils import load_jsonl, dump_jsonl, REPOS_DIR
from tqdm import tqdm

from parso.cache import clear_cache, _get_default_cache_path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node


class JavaTreeSitterWrapper:
    def __init__(self) -> None:
        Language.build_library(
            # Store the library in the `build` directory
            "build/my-languages.so",
            # Include one or more languages
            ["tree-sitter/vender/tree-sitter-java"],
        )

        self.JAVA_LANGUAGE = Language("build/my-languages.so", "java")

        self.parser = Parser()
        self.parser.set_language(self.JAVA_LANGUAGE)

    def get_query(self, attribute_type):
        if attribute_type == 'method':
            query = self.JAVA_LANGUAGE.query("""((method_declaration) @method)""")

        if attribute_type == 'all_method_body':
            query = self.JAVA_LANGUAGE.query("""(method_declaration body: (block) @all_method_block)""")
        
        if attribute_type == 'constructor':
            query = self.JAVA_LANGUAGE.query("""((constructor_declaration) @constructor_declaration)""")

        if attribute_type == 'all_constructor_body':
            query = self.JAVA_LANGUAGE.query("""((constructor_declaration) body:(block)@all_constructor_block)""")

        if attribute_type == 'all_field_declaration':
            query = self.JAVA_LANGUAGE.query("""(field_declaration) @field_declaration""")
        
        if attribute_type == 'class_name':
            query = self.JAVA_LANGUAGE.query("""(class_declaration
                                  name: (identifier) @class_name)""")

        return query

    # take start line as the first non-commented and non-empty line
    def modified_start_line(self, lines):
        for i in range(len(lines)):
            line = lines[i]
            if line and not (line.startswith('/') or line.startswith('*')): # not part of the license text or empty line
                return i

    def get_tree(self, filename):
        """
        obtain parse tree for a file
        """
        file_str = open(filename, encoding="utf8", errors='backslashreplace').read()
        tree = self.parser.parse(bytes(file_str, "utf-8"))
        root_node = tree.root_node
        return root_node

    def join_lines(lst):
        return ''.join(lst)

    def get_string(self, filename, lines_no) -> str:
        '''
        get the string corresponding to the start and end positions in the parse tree
        '''
        lines = open(filename, encoding="utf8", errors='backslashreplace').readlines()
        ret = ''
        for i in lines_no:
            ret += lines[i]
        return ret

    def parse_captures(self, captures, filename):
        text_spans = []
        for capture in captures:
            #capture[1] = property_name
            start, end = capture[0].start_point, capture[0].end_point
            #text = get_string(filename, start, end)
            text_spans.append((start, end))
        return text_spans

    def get_attribute(self, root_node, filename, attribute_type):

        query = self.get_query(attribute_type)
        captures = query.captures(root_node)
        if captures:
            attributes = self.parse_captures(captures, filename)
        else:
            attributes = [((-1, -1), (-1, -1))]
        return attributes

    def declaration_minus_body(self, root_node, method_query, method_body_query):
        m_captures = method_query.captures(root_node)
        mb_captures = method_body_query.captures(root_node)
        if m_captures and mb_captures:
            try:
                assert len(m_captures) == len(mb_captures)
                captures = [(m_captures[i][0].start_point, mb_captures[i][0].start_point) for i in range(len(m_captures))]

            except:
                captures = []
                for i in range(len(m_captures)):
                    for j in range(len(mb_captures)):
                        if m_captures[i][0].end_point != mb_captures[j][0].end_point:
                            continue
                        captures.append((m_captures[i][0].start_point, mb_captures[j][0].start_point))

            attributes = []
            for capture in captures:
                #capture[1] = property_name
                start, end = capture[0], capture[1]
                #text = get_string(filename, start, end)
                attributes.append((start, end))
            # attributes = parse_captures(m_captures, fpath)
        else:
            attributes = [((-1, -1), (-1, -1))]

        return attributes
    

    def get_method_signatures(self, fpath) -> set:
        root_node = self.get_tree(fpath)
        method_query = self.get_query('method')
        method_body_query = self.get_query('all_method_body')

        constructor_query = self.get_query('constructor')
        constructor_body_query = self.get_query('all_constructor_body')

        attributes = self.declaration_minus_body(root_node, method_query, method_body_query)
        attributes.extend(self.declaration_minus_body(root_node, constructor_query, constructor_body_query))

        lines_no = set()
        for t in attributes:
            # t -> ((1, 2), (3, 2))
            for i in range(t[0][0], t[1][0] + 1):
                lines_no.add(i)
        return lines_no
    
    def get_declaration(self, fpath, q_type) -> set:
        root_node = self.get_tree(fpath)
        targets_nodes = self.get_attribute(root_node, fpath, q_type)
        lines_no = set()
        for node in targets_nodes:
            for i in range(node[0][0], node[1][0] + 1):
                lines_no.add(i)
        return lines_no

    def extract_import_info(self, fpath):
        lines_no = self.get_method_signatures(fpath)
        lines_no = lines_no.union(self.get_declaration(fpath, 'all_field_declaration'))
        lines_no = lines_no.union(self.get_declaration(fpath, 'class_name'))
        lines_no_ls = [i for i in lines_no if i >= 0]
        lines_no_ls.sort()
        return self.get_string(fpath, lines_no_ls)

def clear_sa_cache():
    cache_path = _get_default_cache_path()
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    clear_cache()

def add_sys_path(case, repo_dir):
    added_path = []
    filepath = repo_dir + "/" + case["metadata"]["repository"] + "/" + case["metadata"]["file"]
    added_path.append(os.path.dirname(filepath))

    sub_repo_dir = repo_dir + "/" + case["metadata"]["repository"]
    added_path.append(sub_repo_dir)
    added_path.append(sub_repo_dir + "/src")
    added_path.append(sub_repo_dir + "/" + case["metadata"]["repository"])

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
    return added_path

def get_module_path(node, src, module_name, repo_dir, script):
    pattern = fr'\b{module_name}\b'
    try:
        match = re.search(pattern, src)
    except Exception as e:
        print("ERROR", e)

    if match:
        lineno = node.lineno
        colno = match.end()
        modules = script.goto(lineno, colno)
        if modules:
            module_path = modules[0].module_path
            if str(module_path).startswith(repo_dir):
                return str(module_path), modules[0]
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None

def handle_function(modules):
    return "def " + modules[0].get_signatures()[0].to_string()

def handle_module(module, repo_dir, deep=0):
    try:
        deep += 1
        if not str(module.module_path).startswith(repo_dir) or deep > 3:
            return ""
        if module.type == "function":
            try:
                return "def " + module.get_signatures()[0].to_string()
            except Exception as e:
                return module.description
        elif module.type == "class":
            content = [module.description + ":"]
            for name in module.defined_names():
                if name.type == "class" or name.type == "module":
                    for line in handle_module(name, repo_dir, deep).split("\n"):
                        content.append("  " + line)
                else:
                    content.append("  " + handle_module(name, repo_dir, deep))
            content = [c for c in content if not c.strip()==""]
            return "\n".join(content)
        elif module.type == "statement":
            return module.name
        elif module.type == "module":
            content = [module.description]
            for name in module.defined_names():
                if name.type == "class" or name.type == "module":
                    for line in handle_module(name, repo_dir, deep).split("\n"):
                        content.append("  " + line)
                else:
                    content.append("  " + handle_module(name, repo_dir, deep))
            content = [c for c in content if not c.strip()==""]
            return "\n".join(content)
        else:
            return module.name
    except Exception as e:
        print(module)
        print("ERROR OCCURED WHEN HANDLE MODULES:", e)
        return ""



def find_import_sources(src_file_path, script, repo_dir, src_line_len) -> list:
    sources = {}
    with open(src_file_path, 'r') as f:
        code = f.read()
    try:
        tree = ast.parse(code, filename=src_file_path)
    except:
        return sources

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lineno = node.lineno
            if lineno > src_line_len:
                continue

            # print(src_file_path)
            src = code.split("\n")[lineno-1]
            for alias in node.names:
                module_name = alias.name
                path, module = get_module_path(node, src, module_name, repo_dir, script)
                if path != None:
                    sources[module_name] = {
                        "content": handle_module(module, repo_dir),
                        "path": str(path).replace(repo_dir+"/", ""),
                        "type": "module"
                    }
                    # sources[module_name] = {}
                    # sources[module_name]["path"] = path
                    # sources[module_name]["type"] = "module"
        elif isinstance(node, ast.ImportFrom):
            lineno = node.lineno
            if lineno > src_line_len:
                continue

            end_lineno = node.end_lineno
            if lineno == end_lineno:
                src = code.split("\n")[lineno-1]
            else:
                src = code.split("\n")[lineno-1:end_lineno]

            if isinstance(src, list):
                module_path,_ = get_module_path(node, "\n".join(src), node.module, repo_dir, script)
            else:
                module_path,_ = get_module_path(node, src, node.module, repo_dir, script)

            if module_path != None or node.module == None:
                for alias in node.names:
                    name = alias.name
                    if isinstance(src, list):
                        for line_idx, line in enumerate(src):
                            pattern = fr'\b{name}\b'
                            match = re.search(pattern, line)
                            if match:
                                lineno = node.lineno + line_idx
                                colno = match.end()
                                modules = script.goto(lineno, colno)
                                if modules:
                                    module = modules[0]
                                    # print(module)
                                    sources[name] = {
                                        "content": handle_module(module,repo_dir),
                                        "path": str(module_path).replace(repo_dir + "/", ""),
                                        "type": module.type
                                    }
                                else:
                                    # print("NOT FIND THIS MODULE:", name)
                                    # print(module_path)
                                    # print(lineno)
                                    # print(colno)
                                    # print(name)
                                    # print(src_file_path)
                                    # input()
                                    continue
                    else:
                        pattern = fr'\b{name}\b'
                        match = re.search(pattern, src)
                        if match:
                            lineno = node.lineno
                            colno = match.end()
                            modules = script.goto(lineno, colno)
                            if modules:
                                module = modules[0]
                                # print(module)
                                sources[name] = {
                                    "content": handle_module(module, repo_dir),
                                    "path": str(module_path).replace(repo_dir + "/", ""),
                                    "type": module.type
                                }
                            else:
                                # print("NOT FIND THIS MODULE:", name)
                                continue
                        else:
                            # print("NOT MATCH THIS MODULE:", name)
                            continue
    return sources

def import_analysize(case, language = 'python'):
    if language == 'python':
        if os.path.abspath(os.path.dirname(__file__)) in sys.path:
            sys.path.remove(os.path.abspath(os.path.dirname(__file__)))
        repo_dir = "data/repos/cceval"
        added_path = add_sys_path(case, repo_dir)
        sub_repo_dir = repo_dir + "/" + case["metadata"]["repository"]
        project = jedi.Project(sub_repo_dir)
        project.added_sys_path.extend(added_path)
        src = case['metadata']['prefix_src'] + case['prompt']
        fpath = os.path.join(repo_dir, case["metadata"]["repository"], case['metadata']['file'])
        script = jedi.Script(code=src, path=fpath, project=project)
        sources = find_import_sources(fpath, script, sub_repo_dir, len(src.split("\n")))
        
        clear_sa_cache()
    elif language == 'java':
        sources = {}
        wrapper = JavaTreeSitterWrapper()
        repo_dir = os.path.join(REPOS_DIR[language], case['metadata']['repository'])

        abs_fpath = os.path.join(repo_dir, case['metadata']['file'])
        with open(abs_fpath, 'r') as f:
            lines = f.readlines()
        
        package = None
        for line in lines:
            if line.lstrip().startswith('package'):
                pattern = r"package\s+(.*?);"
                matches = re.findall(pattern, line)
                package = matches[0]
                break
        
        if package == None:
            return None

        for line in lines:
            if line.startswith('import '):
                pattern = r"import\s+(.*?);"
                matches = re.findall(pattern, line)
                import_pkg = matches[0]
                pkg_i = abs_fpath.find(package.replace('.', '/'))
                import_uri = os.path.join(abs_fpath[:pkg_i], import_pkg.replace('.', '/') + '.java')

                if os.path.exists(import_uri):
                    # uris.append(import_uri)
                    sources[import_pkg] = {
                        "content": wrapper.extract_import_info(import_uri),
                        "path": import_uri.replace(repo_dir + '/', ''), # case['metadata']['file'],
                        "type": 'class'
                    }
        
    case["sources"] = sources
    return sources

# if __name__ == "__main__":
#     sys.path.remove(os.path.abspath(os.path.dirname(__file__)))
#     repo_dir = "data/repos/cceval"
#     data = load_jsonl("data/datasets/cceval/cceval_2460.jsonl")[76:]
#     for case in tqdm(data):
#         added_path = add_sys_path(case)
#         sub_repo_dir = repo_dir + "/" + case["metadata"]["repository"]
#         cnt = 0
#         project = jedi.Project(sub_repo_dir)
#         project.added_sys_path.extend(added_path)
#         src = case['metadata']['prefix_src'] + case['prompt']
#         fpath = os.path.join(repo_dir, case["metadata"]["repository"], case['metadata']['file'])
#         script = jedi.Script(code=src, path=fpath, project=project)
#         sources = find_import_sources(fpath, script, sub_repo_dir, len(src.split("\n")))
#         case["sources"] = sources
#         clear_sa_cache()
#         dump_jsonl(data, "src/study/import_sources.jsonl")
