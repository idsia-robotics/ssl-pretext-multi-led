from src.config.argument_parser import parse_args
from src.models import get_model, BaseModel
from src.dataset.dataset import get_dataset
import torch
import torch.utils.data
import mlflow
from src.metrics import binary_auc
from statistics import mean
from tqdm import trange

def train_loop(model : BaseModel, train_dataloader, val_dataloader, device, epochs, lr = .001, validation_rate = 10):

    optimizer = model.optimizer(lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 2e-5, -1)
    
    for e in trange(epochs):
        losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()

            image = batch['image'].to(device)
            proj_uvz = batch['proj_uvz'].to(device)
            u_visible = (proj_uvz[:, 0] > 0) & (proj_uvz[:, 0] < 640)
            v_visible = (proj_uvz[:, 1] > 0) & (proj_uvz[:, 1] < 640)
            z_visible = (proj_uvz[:, 2] > 0)
            robot_visible = (u_visible & v_visible & z_visible).float()
            
            out = model.forward(image)
            
            loss = model.loss(out, robot_visible).mean()
            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()

        mlflow.log_metric('train/loss', mean(losses), e)
        mlflow.log_metric('train/lr', lr_schedule.get_last_lr()[0], e)

        lr_schedule.step()

        

        if val_dataloader and (e + 1) % validation_rate == 0:
            preds = []
            trues = []
            losses = []
            for batch in val_dataloader:
                image = batch['image'].to(device)
                proj_uvz = batch['proj_uvz'].to(device)
                u_visible = (proj_uvz[:, 0] > 0) & proj_uvz[:, 0] < 640
                v_visible = (proj_uvz[:, 1] > 0) & proj_uvz[:, 1] < 640
                z_visible = (proj_uvz[:, 2] > 0)
                robot_visible = u_visible & v_visible & z_visible
                
                out = model.forward(image)
                loss = model.loss(robot_visible, out).mean()
                losses.append(loss.item())

                preds.extend(out.flatten().detach().cpu().numpy())
                trues.extend(robot_visible.flatten().detach().cpu().numpy())
            
            auc = binary_auc(preds, trues)
            mlflow.log_metric('validation/presence/auc', auc, e)
            mlflow.log_metric('validation/loss/', mean(losses), e)


def main():
    args = parse_args("train")
    model_cls = get_model(args.model_type)
    model = model_cls(task = args.task).to(args.device)
    train_dataset = get_dataset(args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, num_workers=0)

    """
    Validation data
    """
    if args.validation_dataset:
        validation_dataset = get_dataset(args.validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64, num_workers=0)
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