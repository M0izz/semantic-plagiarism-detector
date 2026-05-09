"""
evaluation
----------
Benchmark and evaluation tools for the Semantic Plagiarism Detection System.

Modules:
    evaluate            Run precision/recall/F1 evaluation against a labelled
                        benchmark dataset.  Compares Sentence Transformer
                        embeddings vs a TF-IDF lexical baseline.

Data:
    benchmark_dataset.json   25 labelled text pairs (10 plagiarized,
                             15 not plagiarized) spanning heavy paraphrases,
                             light paraphrases, same-topic originals,
                             and different-topic negatives.

Usage (from project root):
    python -m evaluation.evaluate
"""
