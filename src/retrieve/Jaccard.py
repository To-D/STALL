import copy
import os
import numpy as np
from retrieve.utils import Tools,FilePathBuilder
from retrieve.build_vector import BagOfWords, BuildVectorWrapper, build_vector
from tqdm import tqdm
from retrieve.make_window import make_window

class CodeSearchWorker():
    def __init__(self, repo_embedding_lines, query_embedding_lines, sim_scorer, max_top_k, language):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_embedding_lines = query_embedding_lines  # list
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        self.language = language
        # self.output_path = output_path
        # self.log_message = log_message

    def _is_context_after_hole(self, repo_embedding_line, query_line):
        hole_fpath_tuple = tuple(query_line['metadata']['fpath_tuple'])
        context_is_not_after_hole = []
        for metadata in repo_embedding_line['metadata']:
            if self.language != 'python':
                fpath = metadata['fpath_tuple'][1:]
            else:
                fpath = metadata['fpath_tuple']

            if tuple(fpath) != hole_fpath_tuple:
                context_is_not_after_hole.append(True)
                continue
            # now we know that the repo line is in the same file as the hole
            if metadata['end_line_no'] <= query_line['metadata']['context_start_lineno']:
                context_is_not_after_hole.append(True)
                continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)

    def _get_next_k_window(self, similar_window):
        metadata = similar_window[0]['metadata']
        # assert metadata[0]['fpath_tuple'][0] == metadata[0]['repo']

        original_code = Tools.read_code(os.path.join(FilePathBuilder.repo_base_dir, *metadata[0]['fpath_tuple']))
        code_lines = original_code.splitlines()
        end_line_no = metadata[0]['end_line_no']
        window_size = metadata[0]['window_size']
        slice_size = metadata[0]['slice_size']
        new_end_line_no = min(end_line_no + window_size // slice_size, len(code_lines))
        new_start_line_no = max(0, new_end_line_no - window_size)
        content_lines = code_lines[new_start_line_no:new_end_line_no]
        return "\n".join(content_lines)

    def _find_top_k_context(self, query_line):
        top_k_context = []
        query_embedding = np.array(query_line['data'][0]['embedding'])
        for repo_embedding_line in self.repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def run(self):
        query_lines_with_retrieved_results = []
        for query_line in self.query_embedding_lines:
            new_line = copy.deepcopy(query_line)
            top_k_context = self._find_top_k_context(new_line)
            top_k_context = top_k_context[-1:-(self.max_top_k+1):-1] # just the top-3
            similar_code = []
            for context in top_k_context:
                new_context = context[0]
                new_context["score"] = context[1]
                new_context["next_k_window"] = self._get_next_k_window(context)
                similar_code.append(new_context)
            new_line['similar_code'] = similar_code
            query_lines_with_retrieved_results.append(new_line)
        return query_lines_with_retrieved_results

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union

def retrieve_jaccard(queries, repo_dir, window_size=20, slice_size=2):
    res = []
    for query in tqdm(queries):
        res += retrieve_jaccard_single(query, repo_dir, window_size, slice_size)
    return res

def retrieve_jaccard_single(query, repo_dir, window_size=20, slice_size=2, language='python'):
    # make query window
    fpath_tuple = tuple(query['metadata']['fpath_tuple'])
    repo = fpath_tuple[0]
    line_no = query['metadata']['line_no']
    original_code = Tools.read_code(os.path.join(repo_dir, '/'.join(fpath_tuple)))
    code_lines = original_code.splitlines()
    context_start_lineno = query['metadata']['context_start_lineno']
    start_line_no = max(context_start_lineno, line_no - window_size + 1)
    window_lines = [i for i in code_lines[start_line_no:line_no]]
    window_lines.append(query["prompt"].split("\n")[-1])
    query["window"] = {
        'context': '\n'.join(window_lines),
        'metadata': {
            'fpath_tuple': fpath_tuple,
            'line_no': line_no,  # line_no starts from 0
            'task_id': query['metadata']['task_id'],
            'start_line_no': start_line_no,
            'end_line_no': line_no,
            'window_size': window_size,
            'context_start_lineno': context_start_lineno,
            'repo': repo
        }
    }

    # vectorize query window
    query["data"] = [{"embedding": Tools.tokenize(query["window"]["context"])}]

    # search
    if language != 'python':
        repo_window_path = FilePathBuilder.repo_windows_path(os.path.join(language, repo), window_size, slice_size)
    else:
        repo_window_path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
    repo_embedding_path = FilePathBuilder.one_gram_vector_path(repo_window_path)

    max_top_k = 3
    try:
        repo_embedding_lines = Tools.load_jsonl(repo_embedding_path)
    except:
        repo_embedding_path = repo_embedding_path.replace(".jsonl", ".pkl")
        repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
    query_embedding_lines = [query]
    worker = CodeSearchWorker(repo_embedding_lines, query_embedding_lines, jaccard_similarity, max_top_k, language)
    res = worker.run()
    del query['data']
    return res