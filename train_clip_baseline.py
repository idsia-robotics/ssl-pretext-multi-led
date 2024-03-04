from src.config.argument_parser import parse_args
from src.models.clip import ClipHead
from src.dataset.dataset import H5Dataset, get_dataset
import torch
import torch.utils.data
import mlflow
from src.metrics import binary_auc, angle_difference, leds_auc
from statistics import mean
from tqdm import trange
import numpy as np


def get_lr_scheduler(schedule_name, optimizer, epochs, lr):
    if schedule_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr / 100, -1)
    elif schedule_name == 'shark':
        return torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                    [
                                                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 2, lr / 100, -1)
                                                    ] * 2,
                                                     [epochs // 2,])
        
def train_loop(model : ClipHead, train_dataloader, val_dataloader, device,
               epochs, supervised_count, unsupervised_count,
               lr = .001, validation_rate = 10,
               checkpoint_logging_rate = 10,
               loss_weights = {'pos' : .2,'dist' : .0,'ori' : .0,'led' : .8},
               lr_schedule = 'cosine'
               ):
    

    optimizer = model.optimizer(lr)

    lr_schedule = get_lr_scheduler(lr_schedule, optimizer, epochs, lr)

    _cuda_weights = {k: torch.tensor([v], device = device) for k, v in loss_weights.items()}
    supervised_count = torch.tensor([supervised_count + 1e-15], device=device)
    unsupervised_count = torch.tensor([unsupervised_count + 1e-15], device=device)

    for e in trange(epochs):
        losses = [0] * len(train_dataloader)
        multiple_led_losses = [[0] * 6, ] * len(train_dataloader)


        preds = []
        trues = []
        losses = [0] * len(train_dataloader)
        p_losses = [0] * len(train_dataloader)
        d_losses = [0] * len(train_dataloader)
        o_losses = [0] * len(train_dataloader)
        led_preds = []
        led_trues = []
        led_visibility = []

        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            image = batch['image'].to(device)
            out = model.forward(image)

            proj_loss, dist_loss, ori_loss = model.loss(batch, out)
            p_losses[batch_i] = proj_loss.detach().item()
            d_losses[batch_i] = dist_loss.detach().item()
            o_losses[batch_i] = ori_loss.detach().item()

            loss = _cuda_weights['pos'] * proj_loss + _cuda_weights['dist'] * dist_loss + _cuda_weights["ori"] * ori_loss
            loss.backward()
            optimizer.step()

            losses[batch_i] = loss.detach().item()
        mlflow.log_metric('train/loss', mean(losses), e)
        mlflow.log_metric('train/loss/coefficients/proj', loss_weights['pos'], e)
        mlflow.log_metric('train/loss/coefficients/dist', loss_weights['dist'], e)
        mlflow.log_metric('train/loss/coefficients/ori', loss_weights['ori'], e)
        mlflow.log_metric('train/loss/coefficients/led', loss_weights['led'], e)
        mlflow.log_metric('train/lr', lr_schedule.get_last_lr()[0], e)
        mlflow.log_metric('train/loss/proj', mean(p_losses), e)
        mlflow.log_metric('train/loss/ori', mean(o_losses), e)
        mlflow.log_metric('train/loss/dist', mean(d_losses), e)

        if e % checkpoint_logging_rate == 0 or e == epochs - 1:
            model.log_checkpoint(e)

        lr_schedule.step()

        

        if val_dataloader and (e % validation_rate == 0 or e == epochs - 1):
            preds = []
            trues = []
            losses = []
            p_losses = []
            d_losses = []
            o_losses = []
            theta_preds = []
            theta_trues = []

            model.eval()

            with torch.no_grad():
                for batch in val_dataloader:
                    image = batch['image'].to(device)
                    
                    out = model.forward(image)
                    p_loss, d_loss, o_loss = model.loss(batch, out)
                    loss = _cuda_weights['pos'] * proj_loss + _cuda_weights['dist'] * dist_loss + _cuda_weights["ori"] * ori_loss
                    losses.append(loss.item())
                    p_losses.append(p_loss.item())
                    d_losses.append(d_loss.item())
                    o_losses.append(o_loss.item())

                    pos_preds = model.predict_pos_from_outs(out)
                    preds.extend(pos_preds)
                    trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())

                    theta_trues.extend(batch["pose_rel"][:, -1])
                    theta_preds.extend(model.predict_orientation_from_outs(out)[0])


            
            errors = np.linalg.norm(np.stack(preds) - np.stack(trues), axis = 1)
            mlflow.log_metric('validation/position/median_error', np.median(errors), e)
            mlflow.log_metric('validation/orientation/mean_error', angle_difference(np.array(theta_preds), np.array(theta_trues)).mean(), e)
            mlflow.log_metric('validation/loss/proj', mean(p_losses), e)
            mlflow.log_metric('validation/loss/ori', mean(o_losses), e)
            mlflow.log_metric('validation/loss/dist', mean(d_losses), e)
            mlflow.log_metric('validation/loss', mean(losses), e)


def main():
    args = parse_args("train")

    model = ClipHead().to(args.device)
    train_dataset = train_dataset = get_dataset(args.dataset, sample_count=args.sample_count, sample_count_seed=args.sample_count_seed, augmentations=True,
                                only_visible_robots=True, compute_led_visibility=False,
                                supervised_flagging=None,
                                supervised_flagging_seed=None,
                                )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers=8, shuffle=True)

    """
    Validation data
    """
    if args.validation_dataset:
        validation_dataset = get_dataset(args.validation_dataset, augmentations=False, only_visible_robots=True, compute_led_visibility=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64, num_workers=8)
    else:
        validation_dataloader = None
    
    
    if args.dry_run:
        for name, val in mlflow.__dict__.items():
            if callable(val):
                val = lambda *args, **kwargs: (None, )
    
    loss_weights = {
        'pos' : args.w_proj,
        'dist' : args.w_dist,
        'ori' : args.w_ori,
        'led' : args.w_led,
    }

    ds_size = len(train_dataset)
    supervised_count = args.labeled_count if args.labeled_count else ds_size
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True

    with mlflow.start_run(experiment_id=args.experiment_id, run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        train_loop(model, train_dataloader, validation_dataloader, args.device,
                   epochs=args.epochs, lr=args.learning_rate, loss_weights = loss_weights,
                   supervised_count=supervised_count,
                   unsupervised_count=ds_size - supervised_count,
                   lr_schedule=args.lr_schedule)
        print(run.info.run_id)


if __name__ == "__main__":
    main()
