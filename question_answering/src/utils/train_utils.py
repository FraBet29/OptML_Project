from transformers import (
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from src.utils.eval_utils import evaluate

def train(
    model, 
    tokenizer,
    optimizer, 
    train_dataloader, 
    val_dataloader,
    val_examples,
    device, 
    learning_rate = 5e-5, 
    num_train_epochs = 3,
    warmup_percent=0.1,
    from_checkpoint = None
):
    
    if from_checkpoint:
        model = AutoModelForQuestionAnswering.from_pretrained(from_checkpoint)
        model.to(device)

    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    total_steps = len(train_dataloader) * num_train_epochs
    warmup_steps = int(total_steps * warmup_percent)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.train()
    best_eval_f1 = 0
    for epoch in range(num_train_epochs):
        running_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'start_positions': batch[3],
                'end_positions': batch[4]
            }
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()
        
        running_loss /= len(train_dataloader)
       
        # eval_loss, em_score, f1_score = evaluate(model, tokenizer, val_dataloader, val_examples, device)
        result = evaluate(model, tokenizer, val_dataloader, val_examples, device)
        print(f"Epoch {epoch+1} | Training loss: {running_loss}")
        # print(f"Epoch {epoch+1} | Eval loss: {eval_loss} | EM: {em_score:.2f} | F1: {f1_score:.2f}")
        print(f"Eval result:\n{result}")

        # if f1_score > best_eval_f1:
        #     best_eval_f1 = f1_score
        #     # Checkpoint the model
        #     model.save_pretrained(f"./checkpoints/version-lr{learning_rate}-epochs{epoch+1}")
        #     print(f"Model saved at epoch {epoch+1}")

    return model