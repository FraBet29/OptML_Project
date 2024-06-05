import torch
import collections

from transformers.data.metrics.squad_metrics import squad_evaluate
from tqdm import tqdm

def get_clean_text(tokens, tokenizer):
    text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(tokens)
        )
    # Clean whitespace
    text = text.strip()
    text = " ".join(text.split())
    return text

def preliminary_predictions(tokens, tokenizer, start_logits, end_logits, n_best):

    # Sort start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

    start_indices = [idx for idx, _ in start_idx_and_logit[:n_best]]
    end_indices = [idx for idx, _ in end_idx_and_logit[:n_best]]

    # Extract the question indices
    sep_token = tokenizer.sep_token_id
    question_indices = [i+1 for i, _ in enumerate(tokens[1:tokens.index(sep_token)])]

    # Keep track of all preliminary predictions
    PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_preds = []
    for start_index in start_indices:
        for end_index in end_indices:
            # throw out invalid predictions
            if (start_index in question_indices) or (end_index in question_indices) or (end_index < start_index):
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index = start_index,
                    end_index = end_index,
                    start_logit = start_logits[start_index],
                    end_logit = end_logits[end_index]
                )
            )
    # Sort prelim_preds in descending score order
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    return prelim_preds

def best_predictions(tokens, tokenizer, start_logits, end_logits, prelim_preds, n_best):
    # Keep track of all best predictions
    BestPrediction = collections.namedtuple(
        "BestPrediction", ["text", "start_logit", "end_logit"]
    )

    n_best_predictions = []
    seen_predictions = []
    for pred in prelim_preds:
        if len(n_best_predictions) >= n_best:
            break
        
        # Non-null answers
        if pred.start_index > 0:
            toks = tokens[pred.start_index : pred.end_index+1]
            text = get_clean_text(toks, tokenizer)

            # If this text has been seen already - skip it
            if text in seen_predictions:
                continue

            # Flag text as being seen
            seen_predictions.append(text)

             # Add this text to a pruned list of the top nbest predictions
            n_best_predictions.append(
                BestPrediction(
                    text=text, 
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit
                    )
                )
    # Add the null prediction
    n_best_predictions.append(
        BestPrediction(
            text="", 
            start_logit=start_logits[0], 
            end_logit=end_logits[0]
            )
        )
    return n_best_predictions

def compute_score_difference(predictions):
    """ Assumes that the null answer is always the last prediction """
    score_null = predictions[-1].start_logit + predictions[-1].end_logit
    score_non_null = predictions[0].start_logit + predictions[0].end_logit
    return score_null - score_non_null

def get_robust_prediction(input_ids, tokenizer, start_logits, end_logits, n_best, null_threshold):
    def to_list(tensor):
        return tensor.detach().cpu().tolist()
    
    # Convert the logits and tokens to lists
    start_logits = to_list(start_logits)
    end_logits = to_list(end_logits)
    tokens = to_list(input_ids)

    prelim_preds = preliminary_predictions(tokens, tokenizer, start_logits, end_logits, n_best)
    n_best_preds = best_predictions(tokens, tokenizer, start_logits, end_logits, prelim_preds, n_best)
    score_difference = compute_score_difference(n_best_preds)

     # if score difference > threshold, return the null answer
    if score_difference > null_threshold:
        return "", score_difference
    else:
        return n_best_preds[0].text, score_difference

def evaluate(
    model,
    tokenizer,
    val_dataloader,
    val_examples,
    device,
    n_best = 10,
    null_threshold = 1.0,
):
    if val_dataloader.batch_size != 1:
        raise RuntimeError("Evaluation dataloader batch size must be 1.")

    model.eval()

    preds = []
    null_odds = []
    for i, batch in tqdm(enumerate(val_dataloader), desc="Evaluation"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'start_positions': batch[3], 'end_positions': batch[4]}
        
        # Pass the data thorugh the model and get the logits
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
        
        #  Get the best predicted answer
        pred_answer, score_difference = get_robust_prediction(
            inputs['input_ids'][0], 
            tokenizer,
            start_logits,
            end_logits,
            n_best, 
            null_threshold
        )
        preds.append(pred_answer)
        null_odds.append(score_difference)

    results = squad_evaluate(
        val_examples,
        preds,
        no_answer_probs=null_odds,
        no_answer_probability_threshold=null_threshold
    )

    return results