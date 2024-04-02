#!/usr/bin/env python3

from torch import nn
import mlflow
import torch

from src.evaluation.encoders.autoencoder import AutoEncoder
from src.data import Dataset
from src import const
from src.evaluation.encoders.diffmap import DiffMap


def fit(model, encoder, optimizer, scheduler, loss, dataloader):
    if const.LOG_REMOTE:
        mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items(
        ) if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval:
                print('-' * 10)
            epoch_loss = torch.empty(1)

            for X, y in dataloader:
                optimizer.zero_grad()

                X, y = X.to(const.DEVICE), y.to(const.DEVICE)
                y_pred = model(encoder.encode(X))

                batch_loss = loss(y_pred, y.unsqueeze(1))
                batch_loss.backward()
                optimizer.step()

                epoch_loss = torch.vstack(
                    [epoch_loss.to(const.DEVICE), batch_loss])
            metrics = {'bce_loss': epoch_loss[1:].mean().item()}
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics:
                    print(f'{key}\t\t: {metrics[key]}')
        print('-' * 10)


def evaluate(classifier, encoder):
    dataloader = torch.utils.data.DataLoader(Dataset('test'),
                                             batch_size=const.BATCH_SIZE)

    n_corr = 0
    for X, y in dataloader:
        with torch.no_grad():
            n_corr += ((classifier(encoder.encode(X.to(const.DEVICE)))
                       > 0.5).to(torch.int) == y.to(const.DEVICE)).sum()

    print(f'Accuracy: {n_corr/len(dataloader.dataset)*100}%')


if __name__ == '__main__':
    dataloader = torch.utils.data.DataLoader(Dataset('train'),
                                             batch_size=const.BATCH_SIZE,
                                             shuffle=True)
    encoder = DiffMap()  # change this!
    classifier = nn.Sequential(nn.Linear(768, 1),
                               nn.Sigmoid()).to(const.DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr=const.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  cycle_momentum=False,
                                                  base_lr=const.LR_BOUNDS[0],
                                                  max_lr=const.LR_BOUNDS[1])
    fit(classifier, encoder, optimizer, scheduler, nn.BCELoss(), dataloader)

    classifier.eval()
    torch.save(classifier.state_dict(), const.MODEL_DIR /
               f'{const.MODEL_NAME}_cls_head.pt')
    evaluate(classifier, encoder)
