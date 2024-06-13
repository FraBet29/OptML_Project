import torch
import collections

from transformers.data.metrics.squad_metrics import squad_evaluate, compute_predictions_logits
from tqdm import tqdm
from src.utils.data_utils import load_and_cache_examples

from torch.utils.data import (
    DataLoader, 
    SequentialSampler
)
from transformers.data.processors.squad import SquadResult
import os

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(
    model,
    tokenizer,
    data_dir,
    device,
    eval_batch_size = 8,
    n_best = 10,
    null_threshold = 1.0,
):
    val_features, val_examples, dataset = load_and_cache_examples(data_dir, tokenizer, evaluate=True, use_cached=True)

    val_sampler = SequentialSampler(dataset)
    val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=eval_batch_size)

    model.eval()
    all_results = []    
    for batch in tqdm(val_dataloader, desc="Evaluating"):

        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
            }
            feature_indices = batch[3]
            outputs = model(**inputs)
        
        for i, feature_index in enumerate(feature_indices):
            val_feature = val_features[feature_index.item()]
            
            unique_id = int(val_feature.unique_id)
            start_logits = outputs.start_logits[i]
            end_logits = outputs.end_logits[i]
            
            result = SquadResult(unique_id, to_list(start_logits), to_list(end_logits))
            all_results.append(result)
    
    # Get the best predicted answer
    predictions = compute_predictions_logits(
        val_examples,
        val_features,
        all_results,
        n_best,
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file=None,
        output_nbest_file=None,
        output_null_log_odds_file=None,
        verbose_logging=True,
        version_2_with_negative=True,
        null_score_diff_threshold=null_threshold,
        tokenizer=tokenizer
    )

    results = squad_evaluate(val_examples, predictions)

    return results