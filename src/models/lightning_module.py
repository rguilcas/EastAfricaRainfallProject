"""
File documentation:
This file defines the RegressionModel class, which is a PyTorch LightningModule for training a regression model.
The model takes a neural network as input and defines the training, validation, testing, and prediction steps.
The module also computes the R2 score on the validation set at the end of each epoch.
"""

import lightning as L
import torch.nn as nn
import torch
from sklearn.metrics import r2_score

class RegressionModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3, target_inverse_transform=None):
        """
        Initializes the RegressionModel class.
        """
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.target_inverse_transform = target_inverse_transform
        self.val_preds = []
        self.val_targets = []
        self.final_val_preds = None
        self.final_val_targets = None

    def forward(self, x):
        """
        Forward pass through the neural network.
        """
        return self.model(x)   
    
    def training_step(self, batch, batch_idx):
        """
        Defines how the training step is performed, including loss computation and logging.
        """
        x, y = batch["x"], batch["y"]
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Log epoch-level mean to avoid noisy last-batch values dominating charts.
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Defines how the validation step is performed, including loss computation and logging.
        """
        x, y = batch["x"], batch["y"]
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_preds.append(y_hat.detach())
        self.val_targets.append(y.detach())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines how the test step is performed, including loss computation and logging.
        """
        x, y = batch["x"], batch["y"]
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Defines how the prediction step is performed, including returning the predictions, targets, and time indices for each batch.
        """
        x, y, time_idx = batch["x"], batch["y"], batch["idx"]
        x = x.float()
        return self(x), y, time_idx

    def on_validation_epoch_start(self):
        """
        Initializes lists to store predictions and targets for the validation epoch.
        """
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self):
        """
        This is applied at the end of each validation epoch. 
        It concatenates the predictions and targets, applies the inverse transformation if necessary, and evaluates the model by computing the R2 score, which is then logged.
        """
        if not self.val_preds:
            return
        y_pred = torch.cat(self.val_preds, dim=0)
        y_true = torch.cat(self.val_targets, dim=0)
        if self.target_inverse_transform is not None:
            y_pred = self.target_inverse_transform(y_pred)
            y_true = self.target_inverse_transform(y_true)

        self.final_val_preds = y_pred.detach().cpu()
        self.final_val_targets = y_true.detach().cpu()

        r2 = r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        self.log("val_r2", r2, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Defines the optimizer to use for training the model. In this case, it uses the AdamW optimizer with the specified learning rate.
        We can customize the optimizer and learning rate scheduler here if needed, and they could be configured from the config file as well.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer