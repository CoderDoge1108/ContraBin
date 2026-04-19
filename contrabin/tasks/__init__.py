"""Downstream tasks for ContraBin.

The paper (TMLR 2025) evaluates four downstream tasks. We keep two of them
essentially unchanged and revise the other two to be better-posed and more
research-relevant:

==========================  ======================================  ==========================================
Task                        Dataset                                 Metrics
==========================  ======================================  ==========================================
Binary retrieval (revised)  POJ-104 (retrieval protocol)            mAP, MRR, Recall@k
Function name recovery      DIRE                                    Exact-match, subtoken F1
Binary summarization        AnghaBench (binary -> NL)               BLEU, ROUGE-L
Compiler provenance (new)   Multi-compiler AnghaBench binaries      Compiler / opt-level / language accuracy
==========================  ======================================  ==========================================

Design notes
------------
* **Binary retrieval** replaces the paper's POJ-104 classification formulation,
  which conflated representation quality with supervised-head capacity. The
  retrieval formulation evaluates the *frozen* ContraBin encoder directly.
* **Compiler provenance** replaces the paper's "reverse engineering"
  (binary -> C) task, which required BLEU over source code - a weak and noisy
  signal. Provenance recovery is a well-posed, deterministic labeling task
  that directly measures how much production context the encoder captures.
"""

from contrabin.tasks.binary_retrieval import (
    LinearProbeClassifier,
    RetrievalResult,
    evaluate_retrieval,
    extract_binary_embeddings,
    train_linear_probe,
)
from contrabin.tasks.compiler_provenance import (
    CompilerProvenanceModel,
    ProvenanceLabelSpace,
    ProvenanceResult,
    train_compiler_provenance,
)
from contrabin.tasks.name_recovery import (
    NameRecoveryModel,
    NameRecoveryResult,
    split_function_name,
    train_name_recovery,
)
from contrabin.tasks.summarization import (
    SummarizationModel,
    SummarizationResult,
    train_summarization,
)

__all__ = [
    "CompilerProvenanceModel",
    "LinearProbeClassifier",
    "NameRecoveryModel",
    "NameRecoveryResult",
    "ProvenanceLabelSpace",
    "ProvenanceResult",
    "RetrievalResult",
    "SummarizationModel",
    "SummarizationResult",
    "evaluate_retrieval",
    "extract_binary_embeddings",
    "split_function_name",
    "train_compiler_provenance",
    "train_linear_probe",
    "train_name_recovery",
    "train_summarization",
]
