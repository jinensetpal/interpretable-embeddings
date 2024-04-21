#!/usr/bin/env python3

from itertools import chain
from torch import nn
import mlflow
import torch

from src.evaluation.encoders.autoencoder import AutoEncoder
from src.data import Dataset
from src import const


def fit(model, encoder, optimizer, scheduler, loss, dataloaders):
    best = {'epoch': -1,
            'parameters': model.state_dict(),
            'loss': torch.inf}

    if const.ONLINE: mse = torch.nn.MSELoss()
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            bce_train_loss = torch.empty(1)
            mse_train_loss = torch.empty(1)
            bce_valid_loss = torch.empty(1)
            mse_valid_loss = torch.empty(1)

            for (X_train, y_train), (X_valid, y_valid) in zip(*dataloaders):
                optimizer.zero_grad()

                X_train, y_train, X_valid, y_valid = [x.to(const.DEVICE) for x in (X_train, y_train, X_valid, y_valid)]
                if const.ONLINE:
                    recon_train, enc = encoder.encode(X_train)
                    y_pred_train = model(enc)
                    recon_valid, enc = encoder.encode(X_valid)
                    y_pred_valid = model(enc)
                else:
                    y_pred_train = model(encoder.encode(X_train))
                    y_pred_valid = model(encoder.encode(X_valid))

                bce_loss_train = loss(y_pred_train, y_train)
                bce_loss_valid = loss(y_pred_valid, y_valid)
                if const.ONLINE:
                    mse_loss_train = mse(recon_train, X_train)
                    mse_loss_valid = mse(recon_valid, X_valid)
                    (bce_loss_train + mse_loss_train).backward()
                else: bce_loss_train.backward()
                optimizer.step()

                bce_train_loss = torch.vstack([bce_train_loss.to(const.DEVICE), bce_loss_train])
                bce_valid_loss = torch.vstack([bce_valid_loss.to(const.DEVICE), bce_loss_valid])
                if const.ONLINE:
                    mse_train_loss = torch.vstack([mse_train_loss.to(const.DEVICE), mse_loss_train])
                    mse_valid_loss = torch.vstack([mse_valid_loss.to(const.DEVICE), mse_loss_valid])
            metrics = {'bce_train_loss': bce_train_loss[1:].mean().item(),
                       'bce_valid_loss': bce_valid_loss[1:].mean().item()}
            if const.ONLINE: metrics.update({'mse_train_loss': mse_train_loss[1:].mean().item(),
                                             'mse_valid_loss': mse_valid_loss[1:].mean().item()})
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if best['loss'] > metrics['bce_valid_loss']:
                best = {'loss': metrics['bce_valid_loss'],
                        'parameters': model.state_dict(),
                        'epoch': epoch + 1}
            elif (epoch + 1 - best['epoch']) > const.EARLY_STOPPING_THRESHOLD: break
        model.load_state_dict(best['parameters'])
        mlflow.log_param('selected_epoch', best['epoch'])
        print('-' * 10)


def evaluate(classifier, encoder):
    dataloader = torch.utils.data.DataLoader(Dataset('test'),
                                             batch_size=const.BATCH_SIZE)

    n_corr = 0
    for X, y in dataloader:
        with torch.no_grad():
            n_corr += ((classifier(encoder.encode(X.to(const.DEVICE))) > 0.5).to(torch.int) == y.to(const.DEVICE)).sum()

    print(f'Accuracy: {n_corr/len(dataloader.dataset)*100}%')


if __name__ == '__main__':
    dataloaders = [torch.utils.data.DataLoader(Dataset(split), batch_size=const.BATCH_SIZE,
                                               shuffle=True) for split in ['train', 'valid']]
    encoder = AutoEncoder(train=const.ONLINE)
    classifier = nn.Sequential(nn.Linear(const.HIDDEN_SIZE, 1),
                               nn.Sigmoid()).to(const.DEVICE)
    optimizer = torch.optim.Adam(chain(encoder.model.parameters(), classifier.parameters()) if const.ONLINE else classifier.parameters(),
                                 lr=const.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  cycle_momentum=False,
                                                  base_lr=const.LR_BOUNDS[0],
                                                  max_lr=const.LR_BOUNDS[1])
    fit(classifier, encoder, optimizer, scheduler, nn.BCELoss(), dataloaders)

    classifier.eval()
    torch.save(classifier.state_dict(), const.MODEL_DIR / f'{const.MODEL_NAME}_cls_head.pt')
    evaluate(classifier, encoder)
