import numpy as np
from scipy.io import wavfile
import samplerate
"""
Note: to install samplerate, run
$ pip install samplerate

Documentation: https://python-samplerate.readthedocs.io/en/latest/

"""


class DataReader:
    def __init__(self, samples_per_beat, beats_per_window):
        """
        Initialize DataReader object
        :param samples_per_beat: number of samples per beat
        :param beats_per_window: number of beats per feature vector (window)

        Construct a DataReader object to convert a WAV file to multiple feature vectors (one feature vector per beat)

        """

        self.beatpath = './The Beatles Annotations/beat/'
        self.chordpath = './The Beatles Annotations/chord/'
        self.recpath = './The Beatles Recordings/'

        self.spb = samples_per_beat
        self.bpw = beats_per_window

    def __call__(self, album, song, pitch, delimiter=' '):
        """
        Convert a song from The Beatles dataset to multiple feature vectors (one feature vector per beat)
        :param album: <'01', '02', '03', '06', '07', '12'>
        :param song: <'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14'>
        :param pitch: <'-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6'>

        :return: ndarray of shape [num_beats - 1, self.spb * self.bpw, 2]

        """
        beatfile = '{}{}/{}.txt'.format(self.beatpath, album, song)
        # chordfile = '{}{}/{}/{}_{}.lab'.format(self.chordpath, album, song, song, pitch)
        recfile = '{}{}/{}/{}_{}.wav'.format(self.recpath, album, song, song, pitch)

        # read WAV file as ndarray
        sr, raw_wav = wavfile.read(recfile)

        # read beat timings, and convert the timings to sample indexes in raw_array
        beat = np.loadtxt(beatfile, delimiter=' ')[:, 0]
        beat = np.int32(sr * beat)

        num_beats = beat.shape[0]

        # re-sample on each beat
        feature_beat = np.zeros([num_beats - 1, self.spb, 2])
        for i in range(num_beats - 1):
            raw = raw_wav[beat[i]:beat[i + 1], :]
            resampled = samplerate.resample(raw, self.spb/raw.shape[0])
            feature_beat[i, :resampled.shape[0], :] = resampled

        # construct feature windows from feature_beat
        feature_window = np.zeros([num_beats - 1, self.spb * self.bpw, 2])
        for i in range(num_beats - 1):
            for j in range(self.bpw):
                if j == 0:
                    feature_window[i, ((j - 1) * self.spb):, :] = feature_beat[i]
                elif i >= j:
                    feature_window[i, ((j - 1) * self.spb):(j * self.spb), :] = feature_beat[i - j]

        return feature_window


class ChordReader:
   raise NotImplementedError


class ChordPitchShifter:
    raise NotImplementedError