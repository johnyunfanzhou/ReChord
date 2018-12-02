import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import ReChordDataset
from utils import SongReader

# For using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path):
    return torch.load(path)


def wav2feature(wavfile_path, offset, bpm, model):
    reader = SongReader(model.spb, model.bpw)
    return reader(wavfile_path, offset, bpm)


def rechord(features, model):
    n = features.shape[0]
    if model.output_size == 25:
        label_placeholder = np.zeros([n, 25])
        test_dataset = ReChordDataset(features, label_placeholder)
        test_loader = DataLoader(test_dataset, batch_size=n, shuffle=False)

        all_predictions = np.empty([0, 25])

        model.eval()
        for i, vbatch in enumerate(test_loader):
            feats, label = vbatch
            feats = np.swapaxes(feats, 1, 2)

            feats = feats.to(device)
            label = label.to(device)

            predictions = model(feats.float()).cpu().detach().numpy()
            all_predictions = np.append(all_predictions, predictions, axis=0)

        return all_predictions

    else:
        raise NotImplementedError


def interpret_label(predictions):
    root = ['C ', 'C#', 'D ', 'D#', 'E ', 'F ', 'F#', 'G ', 'G#', 'A ', 'A#', 'B ']

    chord_list = []
    if predictions.shape[1] == 25:
        for i in range(predictions.shape[0]):
            pos = np.argmax(predictions[i])
            notation = ''
            notation = notation + root[int(pos / 2)]
            if pos % 2 == 1:
                notation = notation + ':min'
            else:
                notation = notation + '    '

            chord_list.append(notation)
    elif predictions.shape[1] == 41:
        raise NotImplementedError
    else:
        raise ValueError('Output prediction size is unexpected.')

    return chord_list


def print_result(chord_list, offset, bpm, bpb):
    for i in range(0, len(chord_list) - bpb, bpb):
        p = '[%.2f]Bar %d: ' % (offset + i * 60 / bpm, int(i / bpb + 1))
        for j in range(bpb):
            p = p + chord_list[i + j] + ' | '
        print(p)


def main(model_path, wavpath, offset, bpm, bpb):
    print('Processing WAV file {} with offset {} at bpm {} ...'.format(wavpath, offset, bpm))
    model = load_model(model_path)
    features = wav2feature(wavpath, offset, bpm, model)
    print('Converted to numpy features. Generating result ...')
    predictions = rechord(features, model)
    chord_list = interpret_label(predictions)
    print_result(chord_list, offset, bpm, bpb)

if __name__ == '__main__':
    while True:
        path = input('Enter path to WAV file: ')  # './test/Hey_Jude.wav'
        if path == 'back':
            continue
        if path == 'exit':
            break
        offset = input('Enter the offset (in seconds): ')   # '0.93227'
        if offset == 'back':
            continue
        if offset == 'exit':
            break
        offset = float(offset)
        bpm = input('Enter the tempo of the song (bpm): ')  # '75'
        if bpm == 'back':
            continue
        if bpm == 'exit':
            break
        bpm = int(bpm)
        bpb = input('Enter number of beats per bar (time signature): ').split('/')[0]   # '4'
        if bpb == 'back':
            continue
        if bpb == 'exit':
            break
        bpb = int(bpb)

        main('./result/config6/preliminary_model19.pt', path, offset, bpm, bpb)
