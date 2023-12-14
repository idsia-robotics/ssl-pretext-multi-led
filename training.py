from src.config.argument_parser import parse_args
from src.models import get_model, BaseModel
from src.dataset.dataset import get_dataset
import torch
import torch.utils.data
import mlflow
from src.metrics import binary_auc
from statistics import mean
from tqdm import trange
import numpy as np

def train_loop(model : BaseModel, train_dataloader, val_dataloader, device, epochs, lr = .001, validation_rate = 10,
               checkpoint_logging_rate = 10):

    optimizer = model.optimizer(lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 2e-5, -1)
    
    for e in trange(epochs):
        losses = []
        p_losses = []
        d_losses = []
        o_losses = []

        preds = []
        trues = []

        for batch in train_dataloader:
            optimizer.zero_grad()

            image = batch['image'].to(device)
            
            out = model.forward(image)
            pos_preds = model.predict_pos_from_out(image, out)
            preds.extend(pos_preds)
            trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())

            
            loss, p_loss, d_loss, o_loss = model.loss(batch, out)
            loss = loss.mean()
            loss.backward()
            losses.append(loss.item())
            p_losses.append(p_loss.item())
            d_losses.append(d_loss.item())
            o_losses.append(o_loss.item())
            
            optimizer.step()

        errors = np.linalg.norm(np.array(preds) - np.array(trues), axis = 1)
        mlflow.log_metric('train/position/median_error', np.median(errors), e)
        mlflow.log_metric('train/loss', mean(losses), e)
        mlflow.log_metric('train/loss/p', mean(p_losses), e)
        mlflow.log_metric('train/loss/o', mean(o_losses), e)
        mlflow.log_metric('train/loss/d', mean(d_losses), e)
        mlflow.log_metric('train/lr', lr_schedule.get_last_lr()[0], e)

        if e % checkpoint_logging_rate == 0 or e == epochs - 1:
            model.log_checkpoint(e)

        lr_schedule.step()

        

        if val_dataloader and e % validation_rate == 0:
            preds = []
            trues = []
            losses = []
            p_losses = []
            d_losses = []
            o_losses = []

            for batch in val_dataloader:
                image = batch['image'].to(device)
                
                out = model.forward(image)
                loss, p_loss, d_loss, o_loss = model.loss(batch, out)
                loss = loss.mean()
                loss.backward()
                losses.append(loss.item())
                p_losses.append(p_loss.item())
                d_losses.append(d_loss.item())
                o_losses.append(o_loss.item())
                pos_preds = model.predict_pos(image)
                preds.extend(pos_preds)
                trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())
            
            errors = np.linalg.norm(np.array(preds) - np.array(trues), axis = 1)
            mlflow.log_metric('validation/position/median_error', np.median(errors), e)
            mlflow.log_metric('validation/loss/p', mean(p_losses), e)
            mlflow.log_metric('validation/loss/o', mean(o_losses), e)
            mlflow.log_metric('validation/loss/d', mean(d_losses), e)
            # auc = binary_auc(preds, trues)
            # mlflow.log_metric('validation/presence/auc', auc, e)
            mlflow.log_metric('validation/loss', mean(losses), e)


def main():
    args = parse_args("train")
    model_cls = get_model(args.model_type)
    model = model_cls(task = args.task).to(args.device)
    train_dataset = get_dataset(args.dataset, sample_count=args.sample_count, sample_count_seed=args.sample_count_seed, augmentations=True,
                                only_visible_robots=args.visible)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, num_workers=8)

    """
    Validation data
    """
    if args.validation_dataset:
        validation_dataset = get_dataset(args.validation_dataset, augmentations=False, only_visible_robots=args.visible)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64, num_workers=8)
    else:
        validation_dataloader = None
    
    
    if args.dry_run:
        return

    with mlflow.start_run(experiment_id=args.experiment_id, run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        train_loop(model, train_dataloader, validation_dataloader, args.device,
                   epochs=args.epochs, lr=args.learning_rate)
        

    

    







if __name__ == "__main__":
    main()