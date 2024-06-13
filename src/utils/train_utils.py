import os
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
    grad_acum_steps = 1,
    save_path=None
):
    save_path = f"checkpoints/{optimizer_name}_15k_{learning_rate}_bs{batch_size}" if not save_path else save_path

    wandb.init(project="optml", name=f"{optimizer.__class__.__name__.lower()}-lr{learning_rate}-bs{batch_size}-epochs{num_train_epochs}")

    wandb.define_metric("train_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="train_step")
    wandb.define_metric("learning_rate", step_metric="train_step")
    wandb.define_metric("eval_exact", step_metric="train_step")
    wandb.define_metric("eval_f1", step_metric="train_step")
    wandb.define_metric("memory_usage", step_metric="epoch")
    wandb.define_metric("epoch_duration_seconds", step_metric="epoch")
    
    _, _, train_dataset = load_and_cache_examples(data_dir, tokenizer, evaluate=False, num_examples=None)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_train_epochs
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
        for i, batch in enumerate(train_dataloader):
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
            # optimizer.step()
            # scheduler.step()
            # model.zero_grad()
            if (i+1) % grad_acum_steps == 0 or i == len(train_dataloader) - 1:
                for param in model.parameters():
                    param.grad /= grad_acum_steps
                
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # Update the loss
            running_loss += loss.item()
            loss_log += loss.item()

            # Log the training loss and the learning rate
            if (curr_train_step+1) % log_steps == 0:
                wandb.log({
                    "train_loss": loss_log / log_steps, 
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "train_step": curr_train_step
                })
                loss_log = 0

            if (curr_train_step + 1) % (steps_per_epoch // 3) == 0:
                result = evaluate(model=model, 
                          tokenizer=tokenizer, 
                          data_dir=data_dir, 
                          device=device)
                print(f"Eval result for training step {curr_train_step}:\n{result}")
                wandb.log({
                    "eval_exact": result["exact"],
                    "eval_f1": result["f1"],
                    "train_step": curr_train_step,
                })

            if (curr_train_step + 1) % 1000 == 0:
                optimizer_name = optimizer.__class__.__name__.lower()
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                model.save_pretrained(
                    f"{save_path}/tmp_checkpoint__step{curr_train_step}"
                )
                print(f"Checkpoint saved at step {curr_train_step}.")

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

        # Log the time and memory usage
        tracemalloc.stop()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)

        # Evaluate the model
        result = evaluate(model=model, 
                          tokenizer=tokenizer, 
                          data_dir=data_dir, 
                          device=device)
        print(f"Eval result:\n{result}")

        # Log the eval metrics
        wandb.log({
            "eval_exact": result["exact"],
            "eval_f1": result["f1"],
            "memory_usage": top_stats[0].size_diff,
            "epoch_duration_seconds": epoch_duration,
            "epoch": epoch+1,
            "train_step": curr_train_step,
        })

        # TODO Possibly change later, once you figure out what f1-score threshold should be
        f1_score = result["f1"]
        if f1_score > best_eval_f1:
            best_eval_f1 = f1_score
            
            optimizer_name = optimizer.__class__.__name__.lower()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            model.save_pretrained(
                f"{save_path}/{optimizer_name}-lr{learning_rate}-bs{batch_size}-ep{epoch+1}"
            )
            print(f"Model saved at epoch {epoch+1}")

    return model