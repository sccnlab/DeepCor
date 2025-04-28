from .models import DeepCor_model
from .utils import r_squared_list, EarlyStopper, Scaler
import torch
import torch.optim as optim 
import numpy as np
import sys

class DeepCorTrainer:
    def __init__(self, latent_dim, in_dim, hidden_dims=[64, 128, 256, 256], lr=0.001, weight_decay=0):
        """
        Initialize the model and optimizer.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Current device: {self.device}")
        sys.stdout.flush()
        self.model = DeepCor_model(in_channels=1, in_dim=in_dim, latent_dim=latent_dim, hidden_dims=hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.early_stopper = EarlyStopper(patience=10, min_delta=0.001)
        self.epoch_num = 200

        # Logs
        self.train_loss_L = []
        self.train_recons_L = []
        self.train_KLD_L = []
        self.val_loss_L = []
        self.val_recons_L = []
        self.val_KLD_L = []

    def train(self, train_loader, val_loader):
        """
        Training the model over epochs. Stop the training if early stopping criteria matched.
        This function is used when the ground truth exist when evaluating the simulated dataset. 
        """
        for epoch in range(self.epoch_num):
            print(f'Epoch {epoch+1}/{self.epoch_num}')
            print('-' * 10)

            train_loss = 0.0
            train_reconstruction_loss = 0.0
            train_KLD = 0.0

            self.model.train()
            for inputs_gm, _, inputs_cf in train_loader:
                inputs_gm = inputs_gm.unsqueeze(1).float().to(self.device)
                inputs_cf = inputs_cf.unsqueeze(1).float().to(self.device)

                self.optimizer.zero_grad()
                outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x = self.model.forward_tg(inputs_gm)
                outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s = self.model.forward_bg(inputs_cf)

                loss = self.model.loss_function(
                    outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x,
                    outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s
                )

                loss['loss'].backward()
                self.optimizer.step()

                train_loss += loss['loss'].item()
                train_reconstruction_loss += loss['Reconstruction_Loss'].item()
                train_KLD += loss['KLD'].item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_gm, _, val_cf = next(iter(val_loader))
                val_gm = val_gm.unsqueeze(1).float().to(self.device)
                val_cf = val_cf.unsqueeze(1).float().to(self.device)

                outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x = self.model.forward_tg(val_gm)
                outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s = self.model.forward_bg(val_cf)

                loss_val = self.model.loss_function(
                    outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x,
                    outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s
                )

                if self.early_stopper.early_stop(loss_val['loss']):
                    print("Early stopping triggered.")
                    break

            # Logging
            n = len(train_loader) * 2
            epoch_train_loss = train_loss / n
            epoch_train_recons = train_reconstruction_loss / n
            epoch_train_KLD = train_KLD / n
            epoch_val_loss = loss_val['loss'].item()
            epoch_val_recons = loss_val['Reconstruction_Loss'].item()
            epoch_val_KLD = loss_val['KLD'].item()

            print(f"Train Loss: {epoch_train_loss:.4f}, Recon: {epoch_train_recons:.4f}, KLD: {epoch_train_KLD:.4f}")
            print(f"Val Loss:   {epoch_val_loss:.4f}, Recon: {epoch_val_recons:.4f}, KLD: {epoch_val_KLD:.4f}\n")

            self.train_loss_L.append(epoch_train_loss)
            self.train_recons_L.append(epoch_train_recons)
            self.train_KLD_L.append(epoch_train_KLD)
            self.val_loss_L.append(epoch_val_loss)
            self.val_recons_L.append(epoch_val_recons)
            self.val_KLD_L.append(epoch_val_KLD)

        print("Training finished.")
        return self.model

    def train_real(self, train_loader, val_loader):
        """
        Training the model over epochs. Stop the training if early stopping criteria matched.
        This function is used when the ground truth not exist when training the real dataset.
        """
        for epoch in range(self.epoch_num):
            print(f'Epoch {epoch+1}/{self.epoch_num}')
            print('-' * 10)

            train_loss = 0.0
            train_reconstruction_loss = 0.0
            train_KLD = 0.0

            self.model.train()
            for inputs_gm, inputs_cf in train_loader:
                inputs_gm = inputs_gm.unsqueeze(1).float().to(self.device)
                inputs_cf = inputs_cf.unsqueeze(1).float().to(self.device)

                self.optimizer.zero_grad()
                outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x = self.model.forward_tg(inputs_gm)
                outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s = self.model.forward_bg(inputs_cf)

                loss = self.model.loss_function(
                    outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x,
                    outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s
                )

                loss['loss'].backward()
                self.optimizer.step()

                train_loss += loss['loss'].item()
                train_reconstruction_loss += loss['Reconstruction_Loss'].item()
                train_KLD += loss['KLD'].item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_gm, val_cf = next(iter(val_loader))
                val_gm = val_gm.unsqueeze(1).float().to(self.device)
                val_cf = val_cf.unsqueeze(1).float().to(self.device)

                outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x = self.model.forward_tg(val_gm)
                outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s = self.model.forward_bg(val_cf)

                loss_val = self.model.loss_function(
                    outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_x,
                    outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s
                )

                if self.early_stopper.early_stop(loss_val['loss']):
                    print("Early stopping triggered.")
                    break

            # Logging
            n = len(train_loader) * 2
            epoch_train_loss = train_loss / n
            epoch_train_recons = train_reconstruction_loss / n
            epoch_train_KLD = train_KLD / n
            epoch_val_loss = loss_val['loss'].item()
            epoch_val_recons = loss_val['Reconstruction_Loss'].item()
            epoch_val_KLD = loss_val['KLD'].item()

            print(f"Train Loss: {epoch_train_loss:.4f}, Recon: {epoch_train_recons:.4f}, KLD: {epoch_train_KLD:.4f}")
            print(f"Val Loss:   {epoch_val_loss:.4f}, Recon: {epoch_val_recons:.4f}, KLD: {epoch_val_KLD:.4f}\n")

            self.train_loss_L.append(epoch_train_loss)
            self.train_recons_L.append(epoch_train_recons)
            self.train_KLD_L.append(epoch_train_KLD)
            self.val_loss_L.append(epoch_val_loss)
            self.val_recons_L.append(epoch_val_recons)
            self.val_KLD_L.append(epoch_val_KLD)

        print("Training finished.")
        return self.model

    def test(self, test_loader):
        """
        Test the model by generating the denoised data.
        This function is used when the ground truth exist when evaluating the simulated dataset. 
        """
        self.model.eval()
        with torch.no_grad():
            test_gm, test_gt, _ = next(iter(test_loader))
            test_gm = test_gm.unsqueeze(1).float().to(self.device)
            test_gt = test_gt.squeeze().cpu().numpy()

            output_test, input_test, fg_mu_z, fg_log_var_z = self.model.forward_fg(test_gm)
            output_np = output_test.squeeze().cpu().numpy()

            # Normalize predictions
            scaler = Scaler(output_np)
            output_scaled = scaler.transform(output_np)

            r2_list = r_squared_list(test_gt, output_scaled)

        return r2_list