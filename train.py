"""
Module for training ResNet
"""


import os
import numpy as np
import sys
import psutil

from torch.utils.data import DataLoader
from torch.autograd import Variable

from resnet import ResNet, ResBlock, _add_conv
from dataset import ImageDataset
from pretrained_model import Net

from torch.optim import lr_scheduler
import torch
import config as cfg
from tqdm import tqdm


class Train:

    def __init__(self) -> None:
        """
        SRresNet and VGGLoss are loaded from related files
        psutil is needed to calculate CPU cores

        """

        # load RresNet or pretrained model
        self.model = None
        pretrained: bool = cfg.PRETRAINED
        pretrained_model_name = cfg.PRETRAINED_MODEL
        num_classes = cfg.NUM_CLASSES

        # for loss only
        self.loss_weights = cfg.LOSS_WEIGHTS

        if pretrained:
            # change last softamx if not NLLLoss
            self.model = Net(pretrained_model_name, num_classes)

        else:
            self.model = ResNet(ResBlock, in_channels=3,
                                make_conv=_add_conv, classes=num_classes)

        # define loss
        # self.loss = torch.nn.CrossEntropyLoss()

        cuda = torch.cuda.is_available()

        if cuda:

            self.model.cuda()
            self.loss_weights = cfg.LOSS_WEIGHTS
            self.loss_weights.cuda()
            self.loss = torch.nn.NLLLoss(weight=self.loss_weights)
            self.loss.cuda()

        # self.loss = torch.nn.NLLLoss(weight=self.loss_weights)

        # from config
        self.learning_rate = cfg.LERANING_RATE  # 3e-4
        self.b1 = cfg.B1  # 0.5
        self.b2 = cfg.B2  # 0.999
        self.train_mean = cfg.TRAIN_MEAN  # np.array([0.485, 0.456, 0.406])
        self.train_std = cfg.TRAIN_STD  # np.array([0.229, 0.224, 0.225])
        self.test_mean = cfg.TEST_MEAN  # np.array([0.485, 0.456, 0.406])
        self.test_std = cfg.TEST_STD  # np.array([0.229, 0.224, 0.225])
        self.n_epoch = cfg.EPOCHS  # 200
        self.batch_size = cfg.BATCH_SIZE  # 4
        self.model_save_path = cfg.MODEL_SAVE_PATH
        self.checkpoint_interval = cfg.CHECKPOINT  # 10
        self.train_dataset_path = cfg.TRAIN_DS_PATH
        self.test_dataset_path = cfg.TEST_DS_PATH
        self.gamma = cfg.GAMMA  # 0.05
        self.max_lr = cfg.MAX_LR  # 0.005
        self.train_csv_path = cfg.TRAIN_CSV_PATH
        self.test_csv_path = cfg.TEST_CSV_PATH
        self.step_size = cfg.STEP_SIZE
        self.need_dict = cfg.DF_TYPE_DICT
        self.test_every = cfg.TEST_STEP

        # define ptimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=self.learning_rate, betas=(self.b1, self.b2), amsgrad=True)

        self.Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        # self.step_size = None
        self.scheduler = None
        self.train_dataloader = None
        self.test_dataloader = None

        self.workers = int(psutil.cpu_count(logical=True)/2)
        # self.workers = 3

    def load_data(self, path: str, train: bool = True) -> DataLoader:
        """
        Custom dataloader
        input:
            path - path to dataset
        output
            Dataloader
        """

        if train:

            self.train_dataloader = DataLoader(
                ImageDataset(
                    path=path,
                    csv_path=self.train_csv_path,
                    mean=self.train_mean,
                    std=self.train_std, need_dict=self.need_dict,
                    train=train),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers
            )

        else:

            self.test_dataloader = DataLoader(
                ImageDataset(
                    path=path,
                    csv_path=self.test_csv_path,
                    mean=self.test_mean,
                    std=self.test_std, need_dict=self.need_dict,
                    train=train),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers
            )

    def train(self, scheduler: str = 'cyclic') -> None:
        """
        method to provide training process
        two shedulers may be choosen
        however, cyclic scheduler does it better
        input:
            schedluer - cyclic or step
        """

        self.load_data(path=self.train_dataset_path, train=True)
        self.load_data(path=self.test_dataset_path, train=False)

        self.step_size = len(self.train_dataloader) * self.step_size

        if scheduler == 'cyclic':

            self.scheduler = lr_scheduler.CyclicLR(self.optimizer,
                                                   base_lr=self.learning_rate,
                                                   max_lr=self.max_lr,
                                                   step_size_up=self.step_size,
                                                   mode='triangular2',
                                                   cycle_momentum=False)

        if scheduler == 'step':

            self.scheduler = lr_scheduler.StepLR(
                self.optimizer, step_size=self.step_size, gamma=self.gamma)

        for epoch in tqdm(range(self.n_epoch)):
            for i, imgs in enumerate(self.train_dataloader):

                # setup model input
                x = Variable(imgs["img"].type(self.Tensor))
                y = Variable(imgs["label"].type(torch.long)).cuda()
                # y = torch.tensor(self.prediction.iloc[idx, :],dtype=torch.long)

                self.optimizer.zero_grad()

                y_pred = self.model(x)  # mb y_pred

                # calc loss
                loss = self.loss(y_pred, y)

                # backward
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # log
                sys.stdout.write(
                    f"[Epoch: {epoch}/{self.n_epoch}] [Batch {i}/{len(self.train_dataloader)}] [loss: {loss.item()}] [lr: {self.optimizer.param_groups[0]['lr']}]\n")

            # calc test loss and acc every nth epoch
            if epoch % self.test_every == 0:

                test_loss, accuracy = self._eval_model()
                sys.stdout.write(
                    f"[Epoch: {epoch}/{self.n_epoch}] [Batch {i}/{len(self.test_dataloader)}] [test_loss: {test_loss}] [acc: {accuracy}]\n")

            # Save model checkpoints
            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                self._save_model(epoch=epoch)

    def _eval_model(self):
        """
        method used in train to evaluate test loss and accuarace
        """

        correct_predictions: int = 0
        accuracy: int = 0

        self.model.eval()  # to inference mode

        with torch.no_grad():
            for imgs in self.test_dataloader:

                x_test = Variable(imgs["img"].type(self.Tensor))
                y_test = Variable(imgs["label"].type(torch.long)).cuda()

                y_pred_test = self.model(x_test)

                test_loss = self.loss(y_pred_test, y_test)
                correct_predictions += test_loss.item()

                ps = torch.exp(y_pred_test)
                _, top_class = ps.topk(1, dim=1)

                equals = top_class == y_test.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        self.model.train()  # back to train
        return correct_predictions/len(self.test_dataloader), accuracy/len(self.test_dataloader)

    def _save_model(self, epoch: int):

        path = os.path.join(self.model_save_path, f'film_resnet_{epoch}.pth')
        torch.save(self.model.state_dict(), path)


# tests
if __name__ == '__main__':

    trainer = Train()
    trainer.train()
