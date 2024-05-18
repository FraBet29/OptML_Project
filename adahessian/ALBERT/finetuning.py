import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    get_linear_schedule_with_warmup, 
    squad_convert_examples_to_features, 
    SquadExample, 
    AlbertTokenizer, 
    AlbertForQuestionAnswering
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import collections
import re
import timeit
import os

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punct(text):
        return re.sub(r'\W', ' ', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punct(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(gt, pred):

    exact_scores = compute_exact(gt, pred)
    f1_scores = compute_f1(gt, pred)

    return exact_scores, f1_scores
    
def train(model, train_dataset, device, learning_rate=5e-5, num_train_epochs=3, train_batch_size=8, from_checkpoint=None):
    
    if from_checkpoint:
        model = AlbertForQuestionAnswering.from_pretrained(from_checkpoint)
        model.to(device)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(num_train_epochs - 1 if from_checkpoint else num_train_epochs):
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'start_positions': batch[3], 'end_positions': batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # checkpoint the model
        model.save_pretrained(f"adahessian/ALBERT/checkpoints/version-lr{learning_rate}-epochs{epoch+1}")
        print(f"Model saved at epoch {epoch+1}")

    return model

def load_and_cache_examples(tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64, evaluate=False):
    
    where = "evaluation" if evaluate else "training"
    # print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    if not os.path.exists(f"adahessian/ALBERT/data/{where}/features.pt"):
        split = 'validation' if evaluate else 'train'
        ex = load_dataset("squad_v2", split=split)
        examples = []

        for item in ex:
            answer_texts = item['answers']['text']
            answer_starts = item['answers']['answer_start']
            examples.append(
                SquadExample(
                    qas_id=item['id'],
                    question_text=item['question'],
                    context_text=item['context'],
                    answer_text=answer_texts[0] if answer_texts else "",
                    start_position_character=answer_starts[0] if answer_starts else None,
                    title="",
                    is_impossible=item['answers']['text'] == [],
                    answers=item['answers']
                )
            )

        features, dataset = squad_convert_examples_to_features(
            examples=examples[:1000], # I use 1/3 of the dataset
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=1
        )

        # save the features and dataset
        torch.save(features, f"adahessian/ALBERT/data/{where}/features.pt")
        torch.save(dataset, f"adahessian/ALBERT/data/{where}/dataset.pt")
        torch.save(examples, f"adahessian/ALBERT/data/{where}/examples.pt")
        print("Features and dataset saved.")

    else:
        features = torch.load(f"adahessian/ALBERT/data/{where}/features.pt")
        dataset = torch.load(f"adahessian/ALBERT/data/{where}/dataset.pt")
        examples = torch.load(f"adahessian/ALBERT/data/{where}/examples.pt")
        print("Features and dataset loaded.")

    # check if they exist
    if not os.path.exists(f"adahessian/ALBERT/data/{where}/features.pt") or not os.path.exists(f"adahessian/ALBERT/data/{where}/dataset.pt"):
        print("Features and dataset not saved or present.")

    return features, examples, dataset

def evaluate(model, tokenizer, dataset, device):
    """Evaluate the model performance on a given dataset."""
    model.eval()
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
    
    em_total = 0
    f1_total = 0
    total_examples = 0
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'start_positions': batch[3], 'end_positions': batch[4]}
        
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        for i in range(inputs["input_ids"].size(0)):
            start_index = torch.argmax(start_logits[i]).item()
            end_index = torch.argmax(end_logits[i]).item()
            
            if start_index <= end_index:
                pred_answer = tokenizer.decode(inputs["input_ids"][i][start_index:end_index+1], skip_special_tokens=True)
            else:
                pred_answer = ""

            ground_truth_answers = tokenizer.decode(inputs["input_ids"][i][inputs["start_positions"][i]:inputs["end_positions"][i]+1], skip_special_tokens=True)
            
            if not ground_truth_answers:
                continue
            
            match_scores, f1_scores = get_raw_scores(ground_truth_answers, pred_answer)

            em_total += match_scores
            f1_total += f1_scores
            total_examples += 1

    em_score = em_total / total_examples
    f1_score = f1_total / total_examples

    print(f"Exact Match (EM): {em_score:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    return em_score

def main():
    model_name_or_path = 'albert-base-v2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    learning_rate = 5e-5
    epochs = 3

    tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path)
    model = AlbertForQuestionAnswering.from_pretrained(model_name_or_path)
    model.to(device)

    _, _, train_dataset = load_and_cache_examples(tokenizer, evaluate=False)
    _, _, val_dataset = load_and_cache_examples(tokenizer, evaluate=True)

    if os.path.exists(os.getcwd() + "/adahessian/ALBERT/checkpoints/version-lr5e-05-epochs3"):
        print("Loading model from checkpoint...")
        model = AlbertForQuestionAnswering.from_pretrained(os.getcwd() + "/adahessian/ALBERT/checkpoints/version-lr5e-05-epochs3", config=model.config)
        model.to(device)
    else:
        print("Start training...")
        model = train(model, train_dataset, device, learning_rate=learning_rate, num_train_epochs=epochs)
        model.save_pretrained(os.getcwd() + "/adahessian/ALBERT/checkpoints/version-lr5e-05-epochs3")

    print("Start evaluation...")
    evaluate(model, tokenizer, val_dataset, device)

if __name__ == "__main__":
    main()
