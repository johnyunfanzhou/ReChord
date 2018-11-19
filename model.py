import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNReChord(nn.Module):
    def __init__(self, samples_per_beat, beats_per_window, config=0, mode='simple'):
        """
        Initialize the CNNReChord model
        :param samples_per_beat: number of samples per beat
        :param beats_per_window: number of beats per feature vector (window)
        :param config:
        :param mode:
        """
        super(CNNReChord, self).__init__()

        self.spb = samples_per_beat
        self.bpw = beats_per_window

        self.config = config

        self.output_size = 25
        if mode == 'advanced':
            self.output_size = 41
            raise NotImplementedError

        # MODEL 0
        if config == 0:
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat/16), stride=int(samples_per_beat/16))
            self.conv2 = nn.Conv1d(12, 24, kernel_size=16, stride=16)
            self.conv3 = nn.Conv1d(24, 48, kernel_size=2)
            size_after_conv = 48 * (beats_per_window - 1)
            self.size_after_conv = size_after_conv
            # MLP layers
            self.fc1 = nn.Linear(size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        else:
            raise NotImplementedError

    def forward(self, features):
        x = features
        if self.config == 0:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.size_after_conv)
            x = self.fc1(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
        else:
            raise NotImplementedError

        return x
