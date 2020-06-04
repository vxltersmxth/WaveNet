"""
Main model of WaveNet
Calculate loss and optimizing
"""
import os

import torch
import torch.optim
from apex import amp
from wavenet.networks import WaveNet as WaveNetModule
AmpOptimizations = ["O0", "O1", "O2", "O3"]

class WaveNet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr=0.002):

        self.net = WaveNetModule(layer_size, stack_size, in_channels, res_channels)

        self.net.cuda()

        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        optim_level = 1
        self.net, self.optimizer = amp.initialize(
            min_loss_scale=1.0,
            models=self.net,
            optimizers=self.optimizer,
            opt_level=AmpOptimizations[optim_level])

        self._prepare_for_gpu()

    @staticmethod
    def _loss():
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if 'state_dict' not in checkpoint:
            state_dict = checkpoint
            checkpoint['epoch'] = 0
        else:
            state_dict = checkpoint['state_dict']
        
        self.net.load_state_dict(state_dict, strict=False)

    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()

    def train(self, inputs, targets):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, timestep, channels]
        :return: float loss
        """
        outputs = self.net(inputs)

        loss = self.loss(outputs.view(-1, self.in_channels),
                         targets.long().view(-1))

        self.optimizer.zero_grad()

        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        #loss.backward()
        self.optimizer.step()

        return loss.item()

    def generate(self, inputs):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        outputs = self.net(inputs)

        return outputs

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.net.state_dict(), model_path)

