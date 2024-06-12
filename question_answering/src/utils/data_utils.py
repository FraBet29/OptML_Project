import os
import re
import torch

from torch.utils.data import Subset
from transformers.data.processors.squad import SquadV2Processor
from transformers import SquadExample, squad_convert_examples_to_features
from datasets import load_dataset

def load_and_cache_examples(
    data_dir,
    tokenizer,
    num_examples=None,
    max_seq_length=384, 
    doc_stride=128, 
    max_query_length=64, 
    evaluate=False,
    use_cached=True
):
    split = "validation" if evaluate else "train"

    if (not use_cached) or (use_cached and (not os.path.exists(f"{data_dir}/{split}/features.pt"))):
        processor = SquadV2Processor()
        if not evaluate:
            examples = processor.get_train_examples("./data/raw_data/", filename="train-v2.0.json")
        else:
            examples = processor.get_dev_examples("./data/raw_data/", filename="dev-v2.0.json")

        examples = examples if not num_examples else examples[:num_examples]

        # Debug: check type
        print(f"Type of examples: {type(examples)}")

        # Debug: Check examples
        # print(f"Number of examples: {len(examples)}")
        # for example in examples[:5]:  # Just print the first 5 examples for inspection
        #     print(f"Example ID: {example.qas_id}")
        #     print(f"Question Text: {example.question_text} (type: {type(example.question_text)})")
        #     print(f"Context Text: {example.context_text} (type: {type(example.context_text)})")

        # Debug: Ensure tokenizer is properly instantiated
        if tokenizer is None:
            raise ValueError("Tokenizer is not provided")

        # Debug: Tokenize a few examples to check for issues
        # for example in examples[:5]:
        #     print("Tokenizing example:")
        #     print(f"Question: {example.question_text}")
        #     print(f"Context: {example.context_text}")
        #     try:
        #         tokens = tokenizer(example.question_text, example.context_text, max_length=max_seq_length)
        #         print(f"Tokens: {tokens}")
        #     except Exception as e:
        #         print(f"Error tokenizing example ID {example.qas_id}: {e}")
        #         raise e

        features, dataset = squad_convert_examples_to_features(
            examples=examples[:num_examples],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training = not evaluate,
            return_dataset="pt",
            threads=1
        )

        if not os.path.exists(f"{data_dir}/{split}"):
            os.makedirs(f"{data_dir}/{split}")
        torch.save(features, f"{data_dir}/{split}/features.pt")
        torch.save(dataset, f"{data_dir}/{split}/dataset.pt")
        torch.save(examples, f"{data_dir}/{split}/examples.pt")
        print(f"Features and dataset saved for {split}.")

    else:
        features = torch.load(f"{data_dir}/{split}/features.pt")
        dataset = torch.load(f"{data_dir}/{split}/dataset.pt")
        examples = torch.load(f"{data_dir}/{split}/examples.pt")
        print(f"Features and dataset loaded for {split}.")

    # num_examples = len(dataset) if num_examples is None else num_examples
    # features = features[:num_examples]
    # examples = examples[:num_examples]
    # dataset = Subset(dataset, range(num_examples))
    return features, examples, dataset