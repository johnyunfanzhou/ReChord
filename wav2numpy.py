import numpy as np
from scipy.io import wavfile
from scipy import signal


class WAVReader:
    def __init__(self, samples_per_beat):
        self.spb = samples_per_beat

    def __call__(self, file_name, bpm=None):
        sample_rate, raw_array = wavfile.read(file_name)
        audio_length = raw_array.shape[0] / sample_rate
        resampled_array = signal.resample(raw_array, int(audio_length * bpm * self.spb / 60))
        return resampled_array

    def get_bpm(self, file_name):

