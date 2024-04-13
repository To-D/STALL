# STALL
We propose a framework STALL+, which supports an extendable and customizable integration of multiple static analysis strategies into the complete pipeline of LLM-based repository-level code completion. In particular, STALL+ can integrate static analysis along the prompting phase (before model inference), the decoding phase (during model inference), and the post-processing phase (after mode inference). Additionally, STALL+ is not only extendable for different static analysis strategies and their combination, but also compatible with RAG techniques.

## Repository Structure
* **src**: the source code of STALL+.
* **data**: the experimental results of `StarCoderBase-7B`, `CodeLlama-7B` and `DeepSeek-Coder-6.7B` on the Python and Java dataset of the `CrossCodeEval` benchmark.

## Hint
*To improve file transfer efficiency, we have compressed the results into a `zip` file. Please click "<u>**View raw**</u>" to download the file and use the `unzip` command to decompress it.*

