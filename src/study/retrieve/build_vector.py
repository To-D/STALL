# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import  os
import tqdm
import itertools
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

from retrieve.utils import Tools, FilePathBuilder, CONSTANTS, load_jsonl

class BagOfWords:
    def __init__(self, input_file):
        self.input_file = input_file

    def build(self, queries = None):
        print(f'building one gram vector for {self.input_file}')
        if queries == None:
            lines = Tools.load_jsonl(self.input_file)
        else:
            lines = queries
        futures = dict()
        with ProcessPoolExecutor(max_workers=48) as executor:
            for line in lines:
                context = line["context"] if 'context' in line else line["prompt"]
                futures[executor.submit(Tools.tokenize, context)] = line

            new_lines = []
            t = tqdm.tqdm(total=len(futures))
            for future in as_completed(futures):
                line = futures[future]
                tokenized = future.result()
                context = line["context"] if 'context' in line else line["prompt"]
                new_lines.append({
                    'context': context,
                    'metadata': line['metadata'],
                    'data': [{'embedding': tokenized}]
                })
                tqdm.tqdm.update(t)
            output_file_path = FilePathBuilder.one_gram_vector_path(self.input_file)
            Tools.dump_jsonl(new_lines, output_file_path)
            print(f"write into {output_file_path}")


class BuildVectorWrapper:
    def __init__(self, benchmark, vector_builder, repos, window_sizes, slice_sizes, language='python'):
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes
        self.vector_builder = vector_builder
        self.benchmark = benchmark
        self.language = language

    def vectorize_repo_windows(self):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                if self.language != 'python':
                    w_path = FilePathBuilder.repo_windows_path(os.path.join(self.language, repo), window_size, slice_size)
                else:
                    w_path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                builder = self.vector_builder(w_path)
                builder.build()

    def vectorize_baseline_and_ground_windows(self, queries = None):
        if queries == None:
            source_file = FilePathBuilder.search_first_window_path(self.benchmark, CONSTANTS.rg, repo, window_size)
        else:
            source_file = None

        for window_size in self.window_sizes:
            for repo in self.repos:
                builder = self.vector_builder(source_file)
                if queries == None:
                    builder.build()
                else:
                    builder.build(queries)

    def vectorize_prediction_windows(self, mode, prediction_path_template):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(window_size=window_size, slice_size=slice_size)
            for repo in self.repos:
                window_path = FilePathBuilder.gen_first_window_path(
                    self.benchmark, mode, prediction_path, repo, window_size
                )
                builder = self.vector_builder(window_path)
                builder.build()

class BuildEmbeddingVector:
    '''
    utilize external embedding model to generate embedding vector
    '''
    def __init__(self, repos, window_sizes, slice_sizes, language='python'):
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes
        self.language = language

    def build_input_file_for_repo_window(self, slice_size):
        lines = []
        for window_size in self.window_sizes:
            for repo in self.repos:
                if self.language != 'python':
                    file_path = FilePathBuilder.repo_windows_path(os.path.join(self.language, repo), window_size, slice_size)
                else:
                    file_path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                loaded_lines = Tools.load_pickle(file_path)
                for line in loaded_lines:
                    lines.append({
                        'context': line['context'],
                        'metadata': {
                            'window_file_path': file_path,
                            'original_metadata': line['metadata'],
                        },})
        return lines

    def build_input_file_search_first_window(self, mode, benchmark):
        lines = []
        for window_size in self.window_sizes:
            for repo in self.repos:
                file_path = FilePathBuilder.search_first_window_path(benchmark, mode, repo, window_size)
                loaded_lines = Tools.load_pickle(file_path)
                for line in loaded_lines:
                    lines.append({
                        'context': line['context'],
                        'metadata': {
                            'window_file_path': file_path,
                            'original_metadata': line['metadata']
                        }})
        return lines

    def build_input_file_for_gen_first_window(self, mode, benchmark, prediction_path):
        lines = []
        for window_size in self.window_sizes:
            for repo in self.repos:
                file_path = FilePathBuilder.gen_first_window_path(benchmark, mode, prediction_path, repo, window_size)
                loaded_lines = Tools.load_pickle(file_path)
                for line in loaded_lines:
                    lines.append({
                        'context': line['context'],
                        'metadata': {
                            'window_file_path': file_path,
                            'original_metadata': line['metadata']
                        }})
        return lines

    @staticmethod
    def place_generated_embeddings(generated_embeddings):
        vector_file_path_to_lines = defaultdict(list)
        for line in generated_embeddings:
            window_path = line['metadata']['window_file_path']
            original_metadata = line['metadata']['original_metadata']
            vector_file_path = FilePathBuilder.ada002_vector_path(window_path)
            vector_file_path_to_lines[vector_file_path].append({
                'context': line['context'],
                'metadata': original_metadata,
                'data': line['data']
            })
        for vector_file_path, lines in vector_file_path_to_lines.items():
            Tools.dump_pickle(lines, vector_file_path)

def build_vector(repo, window_size, slice_size, language='python'):
    # build vector using codextokenizer
    if isinstance(repo, list):
        repos = repo
    else:
        repos = [repo]

    window_sizes = [window_size]
    slice_sizes = [slice_size]

    vectorizer = BagOfWords
    wrapper = BuildVectorWrapper(None, vectorizer, repos, window_sizes, slice_sizes, language)
    wrapper.vectorize_repo_windows()
    print("build vector done")