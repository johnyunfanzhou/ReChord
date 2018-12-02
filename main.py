import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from dataset import ReChordDataset
from model import CNNReChord
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import itertools

# For using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

samples_per_beat = 3072
beats_per_window = 9
encode_length = 25
data = [('01', '01'), ('01', '02'), ('01', '03'), ('01', '04'), ('01', '05'), ('01', '06'), ('01', '07'),
        ('01', '08'), ('01', '09'), ('01', '10'), ('01', '11'), ('01', '12'), ('01', '13'), ('01', '14'),
        ('02', '01'), ('02', '02'), ('02', '03'), ('02', '04'), ('02', '06'), ('02', '07'), ('02', '08'),
        ('02', '09'), ('02', '10'), ('02', '11'), ('02', '12'), ('02', '14'),
        ('03', '01'), ('03', '02'), ('03', '03'), ('03', '04'), ('03', '05'), ('03', '06'), ('03', '07'),
        ('03', '08'), ('03', '09'), ('03', '10'), ('03', '11'), ('03', '12'), ('03', '13'),
        ('06', '01'), ('06', '02'), ('06', '03'), ('06', '04'), ('06', '05'), ('06', '06'), ('06', '07'),
        ('06', '08'), ('06', '09'), ('06', '10'), ('06', '11'), ('06', '12'), ('06', '13'), ('06', '14'),
        ('07', '01'), ('07', '02'), ('07', '03'), ('07', '04'), ('07', '05'), ('07', '06'), ('07', '07'),
        ('07', '08'), ('07', '09'), ('07', '10'), ('07', '11'), ('07', '12'), ('07', '13'), ('07', '14'),
        ('12', '01'), ('12', '02'), ('12', '03'), ('12', '04'), ('12', '05'), ('12', '06'), ('12', '07'),
        ('12', '08'), ('12', '09'), ('12', '10'), ('12', '11'), ('12', '12')]


def load_data(batch_size, album, song):
    features = np.empty([0, samples_per_beat * beats_per_window, 2])
    labels = np.empty([0, encode_length])

    for pitch in ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6']:
        features = np.concatenate((features, np.load('./data/features/{}_{}_{}.npy'.format(album, song, pitch))), axis=0)
        labels = np.concatenate((labels, np.load('./data/labels/{}_{}_{}.npy'.format(album, song, pitch))), axis=0)

    train_data, val_data, train_label, val_label = train_test_split(features, labels, test_size=0.2)

    train_dataset = ReChordDataset(train_data, train_label)
    val_dataset = ReChordDataset(val_data, val_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_data_balanced(batch_size, data_batch_number):
    features = np.load('./data/balanced_features/{}.npy'.format(data_batch_number))
    labels = np.load('./data/balanced_labels/{}.npy'.format(data_batch_number))

    train_data, val_data, train_label, val_label = train_test_split(features, labels, test_size=0.2)

    train_dataset = ReChordDataset(train_data, train_label)
    val_dataset = ReChordDataset(val_data, val_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(lr, config=0, pre_trained=-1):
    if pre_trained >= 0:
        model_file = './result/config{}/preliminary_model{}.pt'.format(config, pre_trained)
        model = torch.load(model_file)
        model.cuda()
        loss_fnc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = CNNReChord(samples_per_beat, beats_per_window, config=config, mode='simple').to(device)
        model.cuda()
        loss_fnc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    model.eval()
    total_correct = 0

    for i, vbatch in enumerate(val_loader):
        feats, labels = vbatch
        feats = np.swapaxes(feats, 1, 2)
        feats = feats.to(device)
        labels = labels.to(device)
        predictions = model(feats.float())
        correct = predictions.argmax(dim=1) == labels.argmax(dim=1)
        total_correct += int(correct.sum())

    return float(total_correct) / len(val_loader.dataset)


def main(batch_size, lr, epochs, config=0, pre_trained=-1):
    model, loss_fnc, optimizer = load_model(lr, config=config, pre_trained=pre_trained)

    for epoch in range(pre_trained + 1, epochs):
        print('Training epoch {} ...'.format(epoch))

        train_accuracy = []
        val_accuracy = []

        # for album, song in data:
        #     train_loader, val_loader = load_data(batch_size, album, song)

        for k in range(60):
            train_loader, val_loader = load_data_balanced(batch_size, k)

            for i, batch in enumerate(train_loader):
                # get the inputs
                feats, label = batch
                feats = np.swapaxes(feats, 1, 2)

                feats = feats.to(device)
                label = label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                predictions = model(feats.float()).squeeze()
                batch_loss = loss_fnc(input=predictions, target=label.argmax(dim=1))

                batch_loss.backward()
                optimizer.step()

            train_acc = evaluate(model, train_loader)
            val_acc = evaluate(model, val_loader)

            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)

            # print('\tFinished album {}, song {} | Training accuracy: {} | Validation accuracy: {}'.format(
            #     album, song, train_acc, val_acc
            # ))

            print('\tFinished data batch number %d | Training accuracy: %.6f | Validation accuracy %.6f' % (
                k, train_acc, val_acc
            ))

        average_train_acc = sum(train_accuracy)/len(train_accuracy)
        average_val_acc = sum(val_accuracy)/len(val_accuracy)
        print('Finished epoch %d | Average training accuracy: %.6f | Average validation accuracy: %.6f' % (
            epoch, average_train_acc, average_val_acc
        ))

        model_file_name = './result/config{}/preliminary_model{}.pt'.format(config, epoch)
        torch.save(model, model_file_name)
        with open('./result/config{}/result.txt'.format(config), 'a+') as file:
            file.write('Preliminary model %d: Average training accuracy: %.4f | Average validation accuracy: %.4f\n' % (
                epoch, average_train_acc, average_val_acc
            ))
        print('Model saved as {}.\n'.format(model_file_name))


if __name__ == "__main__":
    main(64, 0.001, 50, config=8, pre_trained=2)
