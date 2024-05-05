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
from collections import Counter
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

def exact_match_score(prediction, truth):
    return (normalize_answer(prediction) == normalize_answer(truth))

def f1_score(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)
    
def train(model, train_dataset, device, learning_rate=5e-5, num_train_epochs=3, train_batch_size=8):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(num_train_epochs):
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'start_positions': batch[3], 'end_positions': batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model

def load_and_cache_examples(tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64, evaluate=False):
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
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=1
    )
    return features, examples, dataset

def evaluate(model, tokenizer, dataset, device):
    """Evaluate the model performance on a given dataset."""
    model.eval()
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
    
    em_total = 0
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

            match_scores = [exact_match_score(pred_answer, answer) for answer in ground_truth_answers]
            em_total += max(match_scores)
            total_examples += 1

    em_score = em_total / total_examples

    print(f"Exact Match (EM): {em_score:.2f}")

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

    print("Start training...")
    model = train(model, train_dataset, device, learning_rate=learning_rate, num_train_epochs=epochs)
    model.save_pretrained(f"adahessian/ALBERT/checkpoints/version-lr{learning_rate}-epochs{epochs}")

    print("Start evaluation...")
    evaluate(model, tokenizer, val_dataset, device)

if __name__ == "__main__":
    main()
