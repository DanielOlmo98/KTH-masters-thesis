import torch
from unet_model import Unet
from dataloader import CellSegDataset
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

if __name__ == '__main__':
    device = torch.device('cuda:0')
    unet = Unet()
    unet.to(device)
    dataset = CellSegDataset()
    test_size = len(dataset) // 5
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size],
                                         generator=torch.manual_seed(1))
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unet.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(30):
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(train_loader):
            img, gt = data.to(device)
            optimizer.zero_grad()
            output = unet(img)

            loss = loss(output, gt)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.











