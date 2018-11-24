import numpy as np
from scipy.io import wavfile
import samplerate
import csv


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
        chordfile = '{}{}/{}/{}_{}.lab'.format(self.chordpath, album, song, song, pitch)
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
    def __init__(self, mode='simple'):
        """
        Initialize the ChordReader Object
        :param mode: 'simple' for recognizing only major and minor, 'advanced' for more detailed recognition

        """
        self.beat_path = './The Beatles Annotations/beat/'
        self.chord_path = './The Beatles Annotations/chord/'

        self.root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.sh_list = ['maj', 'min', 'dim', 'aug', 'maj7', 'min7', '7', 'dim7', 'hdim7', 'minmaj7', 'maj6', 'min6', '9', 'maj9', 'min9', 'sus4']

        if mode == 'simple':
            self.encode_length = 25
        elif mode == 'advanced':
            self.encode_length = 41
        else:
            raise ValueError('Invalid ChordReader mode. Should be either \'simple\' or \'advanced\'')

    def __call__(self, album, song, pitch=0):
        """
        :param album: <'01', '02', '03', '06', '07', '12'>
        :param song: <'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14'>
        Convert the chord annotation into a ndarray

        :return: ndarray of shape [num_beats - 1, encode_length]

        """
        beat_file = '{}{}/{}.txt'.format(self.beat_path, album, song)
        chord_file = '{}{}/{}.lab'.format(self.chord_path, album, song)
        pitch = pitch % 12

        # read beat timings (from the beat file) and convert it to ndarray
        beat = np.loadtxt(beat_file, delimiter=' ')[:, 0]

        chord = []  # each element in the array are in this form: [start time, end time, root (or 'N'), short_hand (or 'N')]

        # Read the chord file and modify the chord annotations to only include major and minor chords
        raw_array = []
        with open(chord_file) as labfile:
            reader = csv.reader(labfile, delimiter=' ')
            for row in reader:
                raw_array.append(row)

        # load the file to the chord array
        if self.encode_length == 25:
            for i in range(len(raw_array)):
                elem = raw_array[i][2]
                if '(' in elem:
                    elem = elem.split('(')[0]
                if '/' in elem:
                    elem = elem.split('/')[0]
                if ':' in elem:
                    elem = elem.split(':')
                    root = self.root_list.index(elem[0])
                    if elem[1] == '':
                        short_hand = 0
                    else:
                        short_hand = self.sh_list.index(elem[1])
                else:
                    if elem == 'N':
                        root = -1
                        short_hand = -1
                    else:
                        root = self.root_list.index(elem)
                        short_hand = 0

                if short_hand in [0, 3, 4, 6, 10, 12, 13, 15]:
                    short_hand = 0
                elif short_hand in [1, 2, 5, 7, 8, 9, 11, 14]:
                    short_hand = 1
                else:
                    short_hand = -1

                chord.append([float(raw_array[i][0]), float(raw_array[i][1]), root, short_hand])

            # construct the chord_vector where each element of this vector is a one-hot code representing the chord of the beat
            num_beat = np.shape(beat)[0]
            chord_vector = np.zeros([num_beat - 1, self.encode_length])

            a = 0  # a is index in chord array
            tmp = float(chord[a][1])
            for i in range(num_beat - 1):  # i is index as beat number
                while beat[i] > tmp:
                    a += 1
                    tmp = float(chord[a][1])
                root = chord[a][2]
                short_hand = chord[a][3]

                if root == -1 or short_hand == -1:
                    chord_vector[i][-1] = 1
                else:
                    chord_vector[i][2 * root + short_hand] = 1

            # pitch shift the labels of the chord
            chord_vector_shifted = np.zeros(chord_vector.shape)
            chord_vector_shifted[:, 0:(2 * pitch)] = chord_vector[:, (24 - 2 * pitch):24]
            chord_vector_shifted[:, (2 * pitch):24] = chord_vector[:, 0:(24 - 2 * pitch)]
            chord_vector_shifted[:, 24] = chord_vector[:, 24]

            return chord_vector_shifted

        elif self.encode_length == 41:
            raise NotImplementedError

        return

class SongReader:
    def __init__(self, samples_per_beat, beats_per_window):
        """
        Initialize DataReader object
        :param samples_per_beat: number of samples per beat
        :param beats_per_window: number of beats per feature vector (window)

        Construct a DataReader object to convert a WAV file to multiple feature vectors (one feature vector per beat)

        """

        self.spb = samples_per_beat
        self.bpw = beats_per_window

    def __call__(self, path, offset=None, bpm=None):
        """
        Convert a user input song to multiple feature vectors (one feature vector per beat)
        :param path: path to the WAVE audio file
        :param offset: offset of the song
        :param bpm: bpm of the song

        :return: ndarray of shape [num_beats - 1, self.spb * self.bpw, 2]

        """

        if offset is None:
            raise ValueError('Requires offset value.')
        if bpm is None:
            raise ValueError('Requires bpm value.')

        # read WAV file as ndarray
        sr, raw_wav = wavfile.read(path)
        length = raw_wav.shape[0]/sr

        # Calculate the timing based on bpm and offset\
        a = np.arange(length * bpm / 60)                # index of beats
        beat = a * 60 / bpm + offset % (60 / bpm)       # timing of beats adjusted by offset
        if beat[-1] > length:
            beat = beat[:-1]

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
