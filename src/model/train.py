#!/usr/bin/env python3

import mlflow
import torch
import sys

from .vae import Model as VAE
from .ae import Model as AE
from ..data import Dataset
from src import const


def fit(model, optimizer, scheduler, loss, dataloaders):
    best = {'epoch': -1,
            'parameters': model.state_dict(),
            'loss': torch.inf}

    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            mse_train_loss = torch.empty(1)
            mse_valid_loss = torch.empty(1)

            for (X_train, y_train), (X_valid, y_valid) in zip(*dataloaders):
                optimizer.zero_grad()

                X_train, X_valid = [x.to(const.DEVICE) for x in (X_train, X_valid)]
                recon_train, enc = model(X_train)
                with torch.no_grad(): recon_valid, enc = model(X_valid)

                mse_loss_train = loss(recon_train, X_train)
                mse_loss_valid = loss(recon_valid, X_valid)
                mse_loss_train.backward()
                optimizer.step()

                mse_train_loss = torch.vstack([mse_train_loss.to(const.DEVICE), mse_loss_train])
                mse_valid_loss = torch.vstack([mse_valid_loss.to(const.DEVICE), mse_loss_valid])
            metrics = {'mse_train_loss': mse_train_loss[1:].mean().item(),
                       'mse_valid_loss': mse_valid_loss[1:].mean().item()}
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if best['loss'] > metrics['mse_valid_loss']:
                best = {'loss': metrics['mse_valid_loss'],
                        'parameters': model.state_dict(),
                        'epoch': epoch + 1}
            elif (epoch + 1 - best['epoch']) > const.EARLY_STOPPING_THRESHOLD: break
        model.load_state_dict(best['parameters'])
        mlflow.log_param('selected_epoch', best['epoch'])
        print('-' * 10)


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    dataloaders = [torch.utils.data.DataLoader(Dataset(split), batch_size=const.BATCH_SIZE,
                                               shuffle=True) for split in ['train', 'valid']]
    model = AE().to(const.DEVICE) if const.MODEL_NAME.startswith('ae') else VAE().to(const.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  cycle_momentum=False,
                                                  base_lr=const.LR_BOUNDS[0],
                                                  max_lr=const.LR_BOUNDS[1])

    fit(model, optimizer, scheduler, torch.nn.MSELoss(), dataloaders)
    torch.save(model.state_dict(), const.MODEL_DIR / f'{const.MODEL_NAME}.pt')
