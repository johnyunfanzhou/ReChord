import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNReChord(nn.Module):
    def __init__(self, samples_per_beat, beats_per_window, config=0, mode='simple'):
        """
        Initialize the CNNReChord model
        :param samples_per_beat: number of samples per beat
        :param beats_per_window: number of beats per feature vector (window)
        :param config: model configuration. For details, look into the "if config == n:" code
        :param mode: 'simple' for maj and min chords only, 'advanced' for more detailed recognition (not implemented)
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
            """
            conv2 recognizes each beat of music with no overlap
            conv3 recognizes 2-beats of music
            
            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat/16), stride=int(samples_per_beat/16))
            self.conv2 = nn.Conv1d(12, 24, kernel_size=16, stride=16)
            self.conv3 = nn.Conv1d(24, 48, kernel_size=2)
            self.size_after_conv = 48 * (beats_per_window - 1)
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 1:
            """
            conv2 recognizes each beat of music with 1/4 overlap
            conv3 recognizes 6-beats of music
            
            # parameters in convolution layers:  4608 + 4608 + 6912
            # parameters in MLP layers: 86016 + 1600
            
            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat/16), stride=int(samples_per_beat/16))
            self.conv2 = nn.Conv1d(12, 24, kernel_size=16, stride=4)
            self.conv3 = nn.Conv1d(24, 48, kernel_size=6)
            self.size_after_conv = 48 * (4 * beats_per_window - 8)
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 2:
            """
            more output channels to include more patterns (maybe?)
            
            # parameters in convolution layers:  4608 + 9216 + 27648
            # parameters in MLP layers: 172032 + 1600

            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat / 16), stride=int(samples_per_beat / 16))
            self.conv2 = nn.Conv1d(12, 48, kernel_size=16, stride=4)
            self.conv3 = nn.Conv1d(48, 96, kernel_size=6)
            self.size_after_conv = 96 * (4 * beats_per_window - 8)
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 3:
            """
            Four convo layers and even more output channels to include more patterns

            # parameters in convolution layers:  4608 + 9216 + 18432 + 73728
            # parameters in MLP layers: 1327104 + 6400

            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat / 16), stride=int(samples_per_beat / 16))
            self.conv2 = nn.Conv1d(12, 48, kernel_size=16, stride=4)
            self.conv3 = nn.Conv1d(48, 96, kernel_size=4)
            self.conv4 = nn.Conv1d(96, 192, kernel_size=4)
            self.size_after_conv = 192 * (4 * beats_per_window - 9)
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 256)
            self.fc2 = nn.Linear(256, 25)
        elif config == 4:
            """
            What about two convo layers?

            # parameters in convolution layers:  4608 + 3456
            # parameters in MLP layers: 101376 + 6400

            """
            #Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat / 16), stride=int(samples_per_beat / 16))
            self.conv2 = nn.Conv1d(12, 48, kernel_size=16, stride=4)
            self.size_after_conv = 48 * (4 * beats_per_window - 3)
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 5:
            """
            Using pool layers
            
            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 12, kernel_size=int(samples_per_beat / 16))
            self.pool1 = nn.MaxPool1d(16, 16)
            self.conv2 = nn.Conv1d(12, 48, kernel_size=16)
            self.pool2 = nn.MaxPool1d(4, 4)
            self.size_after_conv = 48 * int((int((beats_per_window - 1/16) * samples_per_beat + 1) / 16 - 15) / 4)
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 6:
            """
            No pool on first layer. Increase number of channels.
            
            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 24, kernel_size=int(samples_per_beat / 16), stride=int(samples_per_beat / 16))
            self.conv2 = nn.Conv1d(24, 96, kernel_size=16)
            self.pool2 = nn.MaxPool1d(4, 4)
            self.size_after_conv = 96 * (int((16 * beats_per_window - 15) / 4))
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 7:
            """
            Further increase number of channels.
             
            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 48, kernel_size=int(samples_per_beat / 16), stride=int(samples_per_beat / 16))
            self.conv2 = nn.Conv1d(48, 96, kernel_size=16)
            self.pool2 = nn.MaxPool1d(4, 4)
            self.size_after_conv = 96 * (int((16 * beats_per_window - 15) / 4))
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 64)
            self.fc2 = nn.Linear(64, 25)
        elif config == 8:
            """
            Further increase number of channels.

            """
            # Convolution layers
            self.conv1 = nn.Conv1d(2, 96, kernel_size=int(samples_per_beat / 16), stride=int(samples_per_beat / 16))
            self.conv2 = nn.Conv1d(96, 384, kernel_size=16)
            self.pool2 = nn.MaxPool1d(4, 4)
            self.size_after_conv = 384 * (int((16 * beats_per_window - 15) / 4))
            # MLP layers
            self.fc1 = nn.Linear(self.size_after_conv, 256)
            self.fc2 = nn.Linear(256, 25)
        else:
            raise NotImplementedError

    def forward(self, features):
        x = features
        if self.config in [0, 1, 2]:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.size_after_conv)
            x = self.fc1(x)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
        elif self.config in [3]:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.view(-1, self.size_after_conv)
            x = self.fc1(x)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
        elif self.config in [4]:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(-1, self.size_after_conv)
            x = self.fc1(x)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
        elif self.config in [5]:
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = x.view(-1, self.size_after_conv)
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=1)
        elif self.config in [6, 7, 8]:
            x = F.relu(self.conv1(x))
            x = self.pool2(F.relu(self.conv2(x)))
            x = x.view(-1, self.size_after_conv)
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=1)
        else:
            raise NotImplementedError

        return x
