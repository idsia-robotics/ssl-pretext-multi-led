from src.config.argument_parser import parse_args
from src.models import get_model, BaseModel
from src.dataset.dataset import H5Dataset, get_dataset
import torch
import torch.utils.data
import mlflow
from src.metrics import binary_auc, angle_difference, leds_auc
from statistics import mean
from tqdm import trange
import numpy as np


def train_loop(model : BaseModel, train_dataloader, val_dataloader, device, epochs, lr = .001, validation_rate = 10,
               checkpoint_logging_rate = 10, loss_weights = {'pos' : .2,'dist' : .0,'ori' : .0,'led' : .8}):

    optimizer = model.optimizer(lr)

    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 2e-5, -1)

    lr_schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.ConstantLR(optimizer, .1, total_iters = 5),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-5, -1)
        ],
        milestones=[3,]
    )
    
    for e in trange(epochs):
        losses = []
        p_losses = []
        d_losses = []
        o_losses = []
        led_losses = []
        multiple_led_losses = []


        preds = []
        theta_preds = []
        dist_preds = []

        trues = []
        dist_trues = []
        theta_trues = []

        for batch in train_dataloader:
            optimizer.zero_grad()

            image = batch['image'].to(device)
            
            out = model.forward(image)
            pos_preds = model.predict_pos_from_out(image, out)
            preds.extend(pos_preds)
            trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())
            dist_trues.extend(batch["distance_rel"])
            theta_trues.extend(batch["pose_rel"][:, -1])


            loss, p_loss, d_loss, o_loss, led_loss, m_led_loss = model.loss(batch, out, e,
                                                                            weights = loss_weights)
            loss = loss.mean()
            loss.backward()
            losses.append(loss.item())
            p_losses.append(p_loss.item())
            d_losses.append(d_loss.item())
            o_losses.append(o_loss.item())
            led_losses.append(led_loss.item())
            multiple_led_losses.append([l.item() for l in m_led_loss])

            dpreds = model.predict_dist_from_outs(out)
            tpreds=  model.predict_orientation_from_outs(out)
            theta_preds.extend(tpreds)
            dist_preds.extend(dpreds) 
            optimizer.step()

        errors = np.linalg.norm(np.array(preds) - np.array(trues), axis = 1)
        dist_errors = np.abs(np.array(dist_preds) - np.array(dist_trues))
        multiple_led_losses = np.stack(multiple_led_losses, axis = 0)
        
        mlflow.log_metric('train/position/median_error', np.median(errors), e)
        mlflow.log_metric('train/distance/mean_error', np.mean(dist_errors), e)
        mlflow.log_metric('train/loss', mean(losses), e)
        mlflow.log_metric('train/loss/proj', mean(p_losses), e)
        mlflow.log_metric('train/loss/ori', mean(o_losses), e)
        mlflow.log_metric('train/loss/dist', mean(d_losses), e)
        mlflow.log_metric('train/loss/led', mean(led_losses), e)
        mlflow.log_metric('train/loss/led', mean(led_losses), e)

        mlflow.log_metric('train/loss/coefficienta/proj', loss_weights['pos'], e)
        mlflow.log_metric('train/loss/coefficienta/dist', loss_weights['dist'], e)
        mlflow.log_metric('train/loss/coefficienta/ori', loss_weights['ori'], e)
        mlflow.log_metric('train/loss/coefficienta/led', loss_weights['led'], e)

        for i, led_label, in enumerate(H5Dataset.LED_TYPES):
            mlflow.log_metric(f'train/loss/led/{led_label}', multiple_led_losses[:, i].mean(), e)

        mlflow.log_metric('train/lr', lr_schedule.get_last_lr()[0], e)

        if e % checkpoint_logging_rate == 0 or e == epochs - 1:
            model.log_checkpoint(e)

        lr_schedule.step()

        

        if val_dataloader and e % validation_rate == 0 or e == epochs - 1:
            preds = []
            trues = []
            losses = []
            p_losses = []
            d_losses = []
            o_losses = []
            led_losses = []
            theta_preds = []
            theta_trues = []
            led_preds = []
            led_trues = []

            for batch in val_dataloader:
                image = batch['image'].to(device)
                
                out = model.forward(image)
                loss, p_loss, d_loss, o_loss, led_loss, m_led_loss = model.loss(batch, out, e, weights = loss_weights)
                loss = loss.mean()
                losses.append(loss.item())
                p_losses.append(p_loss.item())
                d_losses.append(d_loss.item())
                o_losses.append(o_loss.item())
                led_losses.append(led_loss.item())

                pos_preds = model.predict_pos_from_outs(image, out)
                preds.extend(pos_preds)
                trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())

                theta_trues.extend(batch["pose_rel"][:, -1])
                theta_preds.extend(model.predict_orientation_from_outs(out))
                led_preds.extend(model.predict_leds_from_outs(out))
                led_trues.extend(batch["led_mask"])

            
            errors = np.linalg.norm(np.stack(preds) - np.stack(trues), axis = 1)
            mlflow.log_metric('validation/position/median_error', np.median(errors), e)
            mlflow.log_metric('validation/orientation/mean_error', angle_difference(np.array(theta_preds), np.array(theta_trues)).mean(), e)
            mlflow.log_metric('validation/loss/proj', mean(p_losses), e)
            mlflow.log_metric('validation/loss/ori', mean(o_losses), e)
            mlflow.log_metric('validation/loss/dist', mean(d_losses), e)
            mlflow.log_metric('validation/loss/led', mean(led_losses), e)

            led_preds = np.array(led_preds)
            led_trues = np.array(led_trues)
            
            led_auc, led_auc_scores = leds_auc(led_preds, led_trues)
            mlflow.log_metric('validation/led/auc', led_auc, e)

            for i, led_label in enumerate(H5Dataset.LED_TYPES):
                mlflow.log_metric(f'validation/led/auc/{led_label}', led_auc_scores[i], e)


            # auc = binary_auc(preds, trues)
            # mlflow.log_metric('validation/presence/auc', auc, e)
            mlflow.log_metric('validation/loss', mean(losses), e)


def main():
    args = parse_args("train")
    model_cls = get_model(args.model_type)
    model = model_cls(task = args.task).to(args.device)
    train_dataset = train_dataset = get_dataset(args.dataset, sample_count=args.sample_count, sample_count_seed=args.sample_count_seed, augmentations=True,
                                only_visible_robots=args.visible, compute_led_visibility=False,
                                supervised_flagging=args.labeled_count,
                                supervised_flagging_seed=args.labeled_count_seed
                                )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, num_workers=8)

    """
    Validation data
    """
    if args.validation_dataset:
        validation_dataset = get_dataset(args.validation_dataset, augmentations=False, only_visible_robots=True, compute_led_visibility=False)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64, num_workers=8)
    else:
        validation_dataloader = None
    
    
    if args.dry_run:
        return
    
    loss_weights = {
        'pos' : args.w_proj,
        'dist' : args.w_dist,
        'ori' : args.w_ori,
        'led' : args.w_led,

    }
    with mlflow.start_run(experiment_id=args.experiment_id, run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        train_loop(model, train_dataloader, validation_dataloader, args.device,
                   epochs=args.epochs, lr=args.learning_rate, loss_weights = loss_weights)
        

if __name__ == "__main__":
    main()
