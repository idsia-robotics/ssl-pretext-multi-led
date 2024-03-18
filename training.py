from src.config.argument_parser import parse_args
from src.models import get_model, BaseModel, load_model_mlflow
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
        
def train_loop(model : BaseModel, train_dataloader, val_dataloader, device,
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
        p_losses = [0] * len(train_dataloader)
        d_losses = [0] * len(train_dataloader)
        o_losses = [0] * len(train_dataloader)
        led_losses = [0] * len(train_dataloader)
        multiple_led_losses = [[0] * 6, ] * len(train_dataloader)


        preds = []
        theta_preds = []
        dist_preds = []

        trues = []
        dist_trues = []
        theta_trues = []

        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            image = batch['image'].to(device)
            out = model.forward(image)

            p_loss, d_loss, o_loss, led_loss, m_led_loss = model.loss(batch, out)

            # same_led_mask = batch["led_mask"][:, -1] == batch["led_mask"][:, -2]
            # p_loss = p_loss[same_led_mask]
            # d_loss = d_loss[same_led_mask]
            # o_loss = o_loss[same_led_mask]
            # led_loss = led_loss[same_led_mask]

            summed_p_loss = p_loss.sum()
            summed_d_loss = d_loss.sum()
            summed_o_loss = o_loss.sum()
            summed_l_loss = led_loss.sum()

            supervised_loss = (
                    _cuda_weights['pos'] * summed_p_loss\
                  + _cuda_weights['dist'] * summed_d_loss \
                  + _cuda_weights['ori'] * summed_o_loss) / supervised_count
            unsupervised_loss = (_cuda_weights['led'] * summed_l_loss) / unsupervised_count
            
            loss = unsupervised_loss + supervised_loss
            loss.backward()
            optimizer.step()

            losses[batch_i] = loss.detach().item()
            p_losses[batch_i] = summed_p_loss.detach().item()
            d_losses[batch_i] = summed_d_loss.detach().item()
            o_losses[batch_i] = summed_o_loss.detach().item()
            led_losses[batch_i] = summed_l_loss.detach().item()
            multiple_led_losses[batch_i] = [l.item() for l in m_led_loss]

            with torch.no_grad():
                pos_preds = model.predict_pos_from_outs(image, out)
                preds.extend(pos_preds)
                trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())
                dist_trues.extend(batch["distance_rel"].cpu().numpy())
                theta_trues.extend(batch["pose_rel"][:, -1].cpu().numpy())
                dpreds = model.predict_dist_from_outs(out).flatten()
                tpreds=  model.predict_orientation_from_outs(out)[0].flatten()
                theta_preds.extend(tpreds)
                dist_preds.extend(dpreds) 

        errors = np.linalg.norm(np.stack(preds) - np.stack(trues), axis = 1)
        dist_errors = np.abs(np.array(dist_preds) - np.array(dist_trues))
        multiple_led_losses = np.stack(multiple_led_losses, axis = 0)
        ori_errors = angle_difference(np.stack(theta_preds), np.stack(theta_trues))
        
        mlflow.log_metric('train/position/median_error', np.median(errors), e)
        mlflow.log_metric('train/distance/median_error', np.median(dist_errors), e)
        mlflow.log_metric('train/orientation/median_error', np.median(ori_errors), e)
        
        mlflow.log_metric('train/loss', sum(losses), e)
        mlflow.log_metric('train/loss/proj', sum(p_losses) / (supervised_count + 1e-15), e)
        mlflow.log_metric('train/loss/ori', sum(o_losses) / (supervised_count + 1e-15), e)
        mlflow.log_metric('train/loss/dist', sum(d_losses) / (supervised_count + 1e-15), e)
        mlflow.log_metric('train/loss/led', sum(led_losses) / (unsupervised_count + 1e-15), e)

        mlflow.log_metric('train/loss/coefficients/proj', loss_weights['pos'], e)
        mlflow.log_metric('train/loss/coefficients/dist', loss_weights['dist'], e)
        mlflow.log_metric('train/loss/coefficients/ori', loss_weights['ori'], e)
        mlflow.log_metric('train/loss/coefficients/led', loss_weights['led'], e)


        for i, led_label, in enumerate(H5Dataset.LED_TYPES):
            mlflow.log_metric(f'train/loss/led/{led_label}', multiple_led_losses[:, i].mean(), e)

        mlflow.log_metric('train/lr', lr_schedule.get_last_lr()[0], e)

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
            led_losses = []
            theta_preds = []
            theta_trues = []
            led_preds = []
            led_trues = []
            led_visibility = []

            model.eval()

            with torch.no_grad():
                for batch in val_dataloader:
                    image = batch['image'].to(device)
                    
                    out = model.forward(image)
                    p_loss, d_loss, o_loss, led_loss, m_led_loss = model.loss(batch, out)
                    mean_p_loss = p_loss.mean().detach()
                    mean_d_loss = d_loss.mean().detach()
                    mean_o_loss = o_loss.mean().detach()
                    mean_l_loss = led_loss.mean().detach()

                    supervised_loss = (
                        loss_weights['pos'] * mean_p_loss\
                    + loss_weights['dist'] * mean_d_loss \
                    + loss_weights['ori'] * mean_o_loss)
                    unsupervised_loss = (loss_weights['led'] * mean_l_loss)
                
                    loss = unsupervised_loss + supervised_loss
                    losses.append(loss.item())
                    p_losses.append(mean_p_loss.item())
                    d_losses.append(mean_d_loss.item())
                    o_losses.append(mean_o_loss.item())
                    led_losses.append(mean_l_loss.item())

                    pos_preds = model.predict_pos_from_outs(image, out)
                    preds.extend(pos_preds)
                    trues.extend(batch['proj_uvz'][:, :-1].cpu().numpy())

                    theta_trues.extend(batch["pose_rel"][:, -1])
                    theta_preds.extend(model.predict_orientation_from_outs(out)[0])
                    led_preds.extend(model.predict_leds(out, batch))
                    led_trues.extend(batch["led_mask"])
                    led_visibility.extend(batch["led_visibility_mask"])

            
            errors = np.linalg.norm(np.stack(preds) - np.stack(trues), axis = 1)
            mlflow.log_metric('validation/position/median_error', np.median(errors), e)
            mlflow.log_metric('validation/orientation/mean_error', angle_difference(np.array(theta_preds), np.array(theta_trues)).mean(), e)
            mlflow.log_metric('validation/loss/proj', mean(p_losses), e)
            mlflow.log_metric('validation/loss/ori', mean(o_losses), e)
            mlflow.log_metric('validation/loss/dist', mean(d_losses), e)
            mlflow.log_metric('validation/loss/led', mean(led_losses), e)

            led_preds = np.array(led_preds)
            led_trues = np.array(led_trues)
            led_visibility = np.array(led_visibility)

            
            aucs = []
            for i, led_label in enumerate(H5Dataset.LED_TYPES):
                vis = led_visibility[:, i]
                # same_led_mask = led_trues[:, -1] == led_trues[:, -2]
                # vis = vis & same_led_mask
                auc = binary_auc(led_preds[vis, i], led_trues[vis, i])
                mlflow.log_metric(f'validation/led/auc/{led_label}', auc, e)
                aucs.append(auc)
            mlflow.log_metric('validation/led/auc', mean(aucs), e)
            mlflow.log_metric('validation/loss', mean(losses), e)


def main():
    args = parse_args("train")

    if args.checkpoint_id:
        model, run_id = load_model_mlflow(experiment_id=args.experiment_id, mlflow_run_name=args.weights_run_name, checkpoint_idx=args.checkpoint_id,
                        model_kwargs={'task' : args.task, 'led_inference' : args.led_inference}, return_run_id=True)
        model = model.to(args.device)
    else:
        model_cls = get_model(args.model_type)
        model = model_cls(task = args.task, led_inference = args.led_inference).to(args.device)

    train_dataset = train_dataset = get_dataset(args.dataset, sample_count=args.sample_count, sample_count_seed=args.sample_count_seed, augmentations=True,
                                only_visible_robots=args.visible, compute_led_visibility=False,
                                supervised_flagging=args.labeled_count,
                                supervised_flagging_seed=args.labeled_count_seed,
                                non_visible_perc=args.non_visible_perc
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
    supervised_count = args.labeled_count if args.labeled_count is not None else ds_size
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
