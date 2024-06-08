import wandb
import time

from transformers import (
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from src.utils.eval_utils import evaluate
from src.utils.data_utils import load_and_cache_examples

from torch.utils.data import (
    DataLoader, 
    RandomSampler
)

import tracemalloc

def train(
    model, 
    tokenizer,
    optimizer, 
    data_dir,
    batch_size,
    device, 
    num_train_epochs = 3,
    warmup_percent=0.1,
    learning_rate=5e-5,
    log_steps = 100,
    from_checkpoint = None
):
    wandb.init(project="optml", name=f"{optimizer.__class__.__name__.lower()}-lr{learning_rate}-epochs{num_train_epochs}")
    

    _, _, train_dataset = load_and_cache_examples(data_dir, tokenizer, evaluate=False, num_examples=5000)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    if from_checkpoint:
        model = AutoModelForQuestionAnswering.from_pretrained(from_checkpoint)
        model.to(device)

    total_steps = len(train_dataloader) * num_train_epochs
    warmup_steps = int(total_steps * warmup_percent)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.train()
    best_eval_f1 = 0
    curr_train_step = 0
    loss_log = 0
    for epoch in range(num_train_epochs):

        tracemalloc.start()
        start_time = time.time()
        snapshot1 = tracemalloc.take_snapshot()
        running_loss = 0

        pbar = tqdm(total=len(train_dataloader))
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'start_positions': batch[3],
                'end_positions': batch[4]
            }
            # Get the model output
            outputs = model(input_ids=inputs['input_ids'], 
                            attention_mask=inputs['attention_mask'], 
                            token_type_ids=inputs['token_type_ids'], 
                            start_positions=inputs['start_positions'], 
                            end_positions=inputs['end_positions'])
            
            # Calculate the loss, backpropagate and update the model
            loss = outputs.loss
            loss.backward(create_graph=True)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            # Update the loss
            running_loss += loss.item()
            loss_log += loss.item()

            # Log the training loss
            if (curr_train_step+1) % log_steps == 0:
                wandb.log({"train_loss": loss_log / log_steps}, step=curr_train_step)
                loss_log = 0
            curr_train_step += 1

            # Update the interactive bar
            pbar.update(1)
            pbar.set_description(f'Train loss: {loss.item():.4f}')

        snapshot2 = tracemalloc.take_snapshot()
        pbar.close()
        epoch_duration = time.time() - start_time
    
        running_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1} | Training loss: {running_loss}")
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = int(epoch_duration % 60)
        print("The duration of the epoch was: ", f"{hours}h {minutes}m {seconds}s")

        tracemalloc.stop()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        wandb.log({
            "memory_usage": top_stats[0].size_diff,
            "epoch_duration_seconds": epoch_duration,
            }, step=curr_train_step)


        result = evaluate(model=model, 
                          tokenizer=tokenizer, 
                          data_dir=data_dir, 
                          device=device)
        print(f"Eval result:\n{result}")

        # Log the eval metrics
        wandb.log({
            "eval_exact": result["exact"],
            "eval_f1": result["f1"],
        }, step=curr_train_step)

        # TODO Possibly change later, once you figure out what f1-score threshold should be
        f1_score = result["f1"]
        if f1_score > best_eval_f1:
            best_eval_f1 = f1_score
            model.save_pretrained(f"./checkpoints/{optimizer.__class__.__name__.lower()}-lr{learning_rate}-bs{batch_size}-ep{epoch+1}")
            print(f"Model saved at epoch {epoch+1}")

    return model