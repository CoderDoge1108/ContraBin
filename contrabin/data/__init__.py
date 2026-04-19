"""Data pipeline: triplet construction, tokenization, and DataLoader factories."""

from contrabin.data.comment_generator import CommentGenerator, HeuristicCommentGenerator
from contrabin.data.compilation import IRCompilationError, compile_c_to_ir
from contrabin.data.datasets import TripletDataset, build_synthetic_triplets
from contrabin.data.loaders import TripletCollator, build_dataloader
from contrabin.data.triplet_builder import TripletBuilder, TripletRecord

__all__ = [
    "CommentGenerator",
    "HeuristicCommentGenerator",
    "IRCompilationError",
    "TripletBuilder",
    "TripletCollator",
    "TripletDataset",
    "TripletRecord",
    "build_dataloader",
    "build_synthetic_triplets",
    "compile_c_to_ir",
]
