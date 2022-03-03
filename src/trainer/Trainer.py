"""
    :filename Trainer.py

    :brief Script containing the training and validation loops.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""
import time
import torch
import config
import logging
import datetime
import LiverTumorDataset
import torch.optim as optim

from tqdm import tqdm
from torch.nn import MSELoss
from torch.optim import AdamW
from torchvision import transforms
from WeightedMSELoss import WeightedMSELoss
from EarlyStopping import EarlyStopping


class Trainer:

    def __init__(self, network, network_name, device, dataset_path, epochs, batch_size, weight_decay, betas, adam_w_eps,
                 early_stopping, lr, lr_scheduler_patience, lr_scheduler_min_lr, lr_scheduler_factor, w_liver, w_tumor):

        self.network = network
        self.network_name = network_name
        self.device = device
        self.dataset_path = dataset_path

        self.epochs = epochs
        self.batch_size = batch_size

        self.weight_decay = weight_decay
        self.betas = betas
        self.adam_w_eps = adam_w_eps
        self.early_stopping_flag = early_stopping

        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.lr_scheduler_factor = lr_scheduler_factor

        self.w_liver = w_liver
        self.w_tumor = w_tumor

        self.criterion = WeightedMSELoss(w_liver=self.w_liver,
                                         w_tumor=self.w_tumor)

        self.optimizer = AdamW(params=self.network.parameters(),
                               lr=self.lr,
                               betas=self.betas,
                               eps=self.adam_w_eps,
                               weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              verbose=True,
                                                              patience=self.lr_scheduler_patience,
                                                              min_lr=self.lr_scheduler_min_lr,
                                                              factor=self.lr_scheduler_factor)

        self.train_loader, self.val_loader = LiverTumorDataset.get_dataset_loaders(dataset_dir=self.dataset_path,
                                                                                   batch_size=self.batch_size,
                                                                                   transforms=transforms.Compose(  # @Lakoc
                                                                                       [transforms.ToTensor],
                                                                                   ))

        if self.early_stopping_flag:
            self.early_stopping = EarlyStopping(patience=config.HYPERPARAMETERS['early_stopping_patience'],
                                                path=f'../../trained-weights/{self.network_name}/best-weights.pt',
                                                verbose=True)

            self.stop_flag = False

        self.training_run_id = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

        self.running_train_loss = 0.0
        self.running_val_loss = 0.0
        self.training_loss_list = []
        self.validation_loss_list = []

        self.scaler = torch.cuda.amp.GradScaler()  # This allows us to use more memory, read docs if interested.

        logging.info(f'Training run ID: {self.training_run_id}.')

        # TODO: plot training and validation loss...

    def training(self):
        start_time = time.time()

        logging.info('Training starts...')

        for epoch in range(1, self.epochs + 1):
            logging.info('Epoch {}/{}'.format(epoch, self.epochs))
            logging.info('-' * 10)

            self.epoch_train(epoch)

            self.epoch_validate(epoch)

            if self.early_stopping_flag and self.stop_flag:  # We are overfitting, let's end training...
                break

        # Load weights from checkpoint: those are the best ones before overfitting.
        self.network.load_state_dict(torch.load(f'../../trained-weights/{self.network_name}/best-weights.pt'))

        name = self.training_run_id + f'-{self.network_name}.pt'
        torch.save(self.network.state_dict(), f'../../trained-weights/{self.network_name}/{name}')
        time_elapsed = time.time() - start_time

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')

    def epoch_train(self, epoch_id):
        self.network.train()

        loop = tqdm(self.train_loader, total=len(self.train_loader))
        for i_batch, sample in enumerate(loop):
            inputs = sample['images'].type(torch.FloatTensor).to(self.device)
            masks = sample['masks'].type(torch.FloatTensor).to(self.device)

            '''Forward. '''
            with torch.cuda.amp.autocast():
                predictions = self.network(inputs)
                loss = self.criterion(predictions, masks).to(self.device)  # Find the loss for the current step

            ''' Backward. '''
            self.scaler.scale(loss / 4).backward()  # Calculate the gradients

            if (i_batch + 1) % 4 == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()  # Clear all the gradients before calculating them

            self.running_train_loss += loss.item()

            loop.set_description(f'Training Epoch {epoch_id}/{self.epochs}')
            loop.set_postfix(loss=loss.item() * self.batch_size)

        self.training_loss_list.append(self.running_train_loss)

    def epoch_validate(self, epoch_id):
        self.network.eval()

        loop = tqdm(self.val_loader, total=len(self.val_loader))
        losses_for_scheduler = []

        with torch.no_grad():
            for i_batch, sample in enumerate(loop):
                inputs = sample['images'].type(torch.FloatTensor).to(self.device)
                masks = sample['masks'].type(torch.FloatTensor).to(self.device)

                predictions = self.network(inputs)

                loss = self.criterion(predictions, masks)

                self.running_val_loss += loss.item()
                losses_for_scheduler.append(loss.item())

                loop.set_description(f'Validating Epoch {epoch_id}/{self.epochs}')
                loop.set_postfix(loss=loss.item() * self.batch_size)

            mean_loss = sum(losses_for_scheduler) / len(losses_for_scheduler)
            self.scheduler.step(mean_loss)

            self.early_stopping(mean_loss, self.network)

            if self.early_stopping_flag and self.early_stopping.early_stop:
                print('Early stopping')
                self.stop_flag = True

        self.validation_loss_list.append(self.running_val_loss)