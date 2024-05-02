#!/usr/bin/env python3

from torcheval.metrics.functional import binary_f1_score
from itertools import chain
from torch import nn
import mlflow
import torch

from src.evaluation.encoders.autoencoder import AutoEncoder
from src.model.loss import VAELoss
from src.data import Dataset
from src import const


def fit(model, encoder, optimizer, scheduler, loss, dataloaders):
    best = {'epoch': -1,
            'parameters': model.state_dict(),
            'loss': torch.inf}

    if const.ONLINE:
        is_ae = const.MODEL_NAME.startswith('ae')
        aux_loss = torch.nn.MSELoss() if is_ae else VAELoss()
        loss_name = 'mse' if is_ae else 'vae'

    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            bce_train_loss = torch.empty(1)
            aux_train_loss = torch.empty(1)
            bce_valid_loss = torch.empty(1)
            aux_valid_loss = torch.empty(1)

            for (X_train, y_train), (X_valid, y_valid) in zip(*dataloaders):
                optimizer.zero_grad()

                X_train, y_train, X_valid, y_valid = [x.to(const.DEVICE) for x in (X_train, y_train, X_valid, y_valid)]
                if const.ONLINE:
                    recon_train, enc = encoder.encode(X_train)
                    y_pred_train = model(enc)
                    with torch.no_grad():
                        recon_valid, enc = encoder.encode(X_valid)
                        y_pred_valid = model(enc)
                else:
                    y_pred_train = model(encoder.encode(X_train))
                    with torch.no_grad(): y_pred_valid = model(encoder.encode(X_valid))

                bce_loss_train = loss(y_pred_train, y_train)
                bce_loss_valid = loss(y_pred_valid, y_valid)
                if const.ONLINE:
                    aux_loss_train = aux_loss(recon_train, X_train)
                    aux_loss_valid = aux_loss(recon_valid, X_valid)
                    (bce_loss_train + aux_loss_train).backward()
                else: bce_loss_train.backward()
                optimizer.step()

                bce_train_loss = torch.vstack([bce_train_loss.to(const.DEVICE), bce_loss_train])
                bce_valid_loss = torch.vstack([bce_valid_loss.to(const.DEVICE), bce_loss_valid])
                if const.ONLINE:
                    aux_train_loss = torch.vstack([aux_train_loss.to(const.DEVICE), aux_loss_train])
                    aux_valid_loss = torch.vstack([aux_valid_loss.to(const.DEVICE), aux_loss_valid])
            scheduler.step()
            metrics = {'lr': optimizer.param_groups[0]['lr'],
                       'bce_train_loss': bce_train_loss[1:].mean().item(),
                       'bce_valid_loss': bce_valid_loss[1:].mean().item()}
            if const.ONLINE: metrics.update({f'{loss_name}_train_loss': aux_train_loss[1:].mean().item(),
                                             f'{loss_name}_valid_loss': aux_valid_loss[1:].mean().item()})
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if best['loss'] > metrics['bce_valid_loss']:
                best = {'loss': metrics['bce_valid_loss'],
                        'parameters': model.state_dict(),
                        'epoch': epoch + 1}
            elif epoch + 1 > const.MIN_EPOCHS and (epoch + 1 - best['epoch']) > const.EARLY_STOPPING_THRESHOLD: break
        model.load_state_dict(best['parameters'])
        mlflow.log_param('selected_epoch', best['epoch'])
        print('-' * 10)


def evaluate(classifier, encoder):
    dataloader = torch.utils.data.DataLoader(Dataset('test'), batch_size=const.BATCH_SIZE)

    net_y, net_pred = torch.empty(1), torch.empty(1)
    for X, y in dataloader:
        with torch.no_grad():
            net_pred = torch.vstack([net_y, classifier(encoder.encode(X.to(const.DEVICE))).detach().cpu()])
            net_y = torch.vstack([net_y, y])

    print(f'F1: {binary_f1_score(net_pred[1:].squeeze(1), net_y[1:].squeeze(1))}')
    print(f'Accuracy: {(net_y == (net_pred > 0.5)).sum() / net_y.shape[0] * 100}%')



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
    if const.ONLINE:
        encoder.model.eval()
        torch.save(encoder.model.state_dict(), const.MODEL_DIR / f'{const.MODEL_NAME}.pt')
    torch.save(classifier.state_dict(), const.MODEL_DIR / f'{const.MODEL_NAME}_cls_head.pt')
    evaluate(classifier, encoder)
