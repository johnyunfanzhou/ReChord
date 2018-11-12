import csv
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
    def __init__(self, encode_length):
        '''
        Initialize the ChordReader Object
        :param encode_length: length of the one-hot encoding of chord, 24 for only major/min chords, 40 for all chords
        '''
        self.beat_path = './The Beatles Annotations/beat/'
        self.chord_path = './The Beatles Annotations/chord/'
        self.encode_length = encode_length

    def __call__(self, album, song, pitch=0):
        '''
        :param album: <'01', '02', '03', '06', '07', '12'>
        :param song: <'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14'>
        Convert the chord annotation into a ndarray
        return each element in the array is a 24-bit vector, represent the chord of the this beat
        '''
        beat_file = '{}{}/{}.txt'.format(self.beat_path, album, song)
        chord_file = '{}{}/{}.lab'.format(self.chord_path, album, song)
        
        # read beat file and convert it to ndarray 
        beat = np.loadtxt(beat_file, delimiter=' ')[:, 0]

        chord = [] # each element in the array are in this form: [start time, end time, chord, major/minor: 1/0]
        tmp_array = []

        # Read the chord file and mordify the chord annotations to only include major and minor chords
        with open(chord_file) as data:
            reader = csv.reader(data, delimiter=' ')
            for row in reader:
                tmp_array.append(row)

        for i in range(len(tmp_array)):
            chord.append([0, 0, 0, 0])

            elem = tmp_array[i][2]
            if '(' in elem:
                elem = elem.split('(')[0]
            if '/' in elem:
                elem = elem.split('/')[0]

            if ':' not in elem:
                chord[i] = [tmp_array[i][0], tmp_array[i][1], elem, 1]
            else:
                pair = elem.split(':')
                root = pair[0]
                short_hand = pair[1]
                
                if (short_hand == 'min') or (short_hand == 'dim') or (short_hand == 'min7') or (short_hand == 'dim7') or (short_hand == 'hdim7') or (short_hand == 'minmaj7') or (short_hand == 'min6') or (short_hand == 'min9'):
                    chord[i] = [tmp_array[i][0], tmp_array[i][1], root, 0]
                elif (short_hand == 'aug') or (short_hand == 'maj7') or (short_hand == '7') or (short_hand == 'maj6') or (short_hand == 'maj9') or (short_hand == '9') or (short_hand == 'sus4'):
                    chord[i] = [tmp_array[i][0], tmp_array[i][1], root, 1]
                else:
                    chord[i] = [tmp_array[i][0], tmp_array[i][1], root, -1]
        
        # construct the chord_vector where each element of this vector is a one-hot code representing the chord of the beat
        num_beat = np.shape(beat)[0]
        chord_vector = np.zeros([num_beat - 1, self.encode_length])

        a = 0
        tmp = float(chord[a][1])
        for i in range(num_beat - 1):
            while beat[i] > tmp:
                a += 1
                tmp = float(chord[a][1])
            if chord[a][2] == "C":
                if chord[a][3] == 1:
                    chord_vector[i][0] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][1] = 1
            elif chord[a][2] == "C#":
                if chord[a][3] == 1:
                    chord_vector[i][2] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][3] = 1
            elif chord[a][2] == "D":
                if chord[a][3] == 1:
                    chord_vector[i][4] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][5] = 1
            elif chord[a][2] == "D#":
                if chord[a][3] == 1:
                    chord_vector[i][6] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][7] = 1
            elif chord[a][2] == "E":
                if chord[a][3] == 1:
                    chord_vector[i][8] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][9] = 1
            elif chord[a][2] == "F":
                if chord[a][3] == 1:
                    chord_vector[i][10] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][11] = 1
            elif chord[a][2] == "F#":
                if chord[a][3] == 1:
                    chord_vector[i][12] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][13] = 1
            elif chord[a][2] == "G":
                if chord[a][3] == 1:
                    chord_vector[i][14] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][15] = 1
            elif chord[a][2] == "G#":
                if chord[a][3] == 1:
                    chord_vector[i][16] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][17] = 1
            elif chord[a][2] == "A":
                if chord[a][3] == 1:
                    chord_vector[i][18] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][19] = 1
            elif chord[a][2] == "A#":
                if chord[a][3] == 1:
                    chord_vector[i][20] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][21] = 1
            elif chord[a][2] == "B":
                if chord[a][3] == 1:
                    chord_vector[i][22] = 1
                elif chord[a][3] == 0:
                    chord_vector[i][23] = 1
            else:
                continue

        # pitch shift the labels of the chord
        if pitch != 0:
            chord_vector_shifted = np.zeros(chord_vector.shape)
            chord_vector_shifted[:, (2 * pitch):] = chord_vector[:, :(-2 * pitch)]
            chord_vector_shifted[:, :(2 * pitch)] = chord_vector[:, (-2 * pitch):]
            chord_vector = chord_vector_shifted

        return chord_vector
