
class Trainer:
    def __init__(self):
        pass

    def train(self):
        model.train()
        criterion = nn.CrossEntropyLoss(reduction="mean")
        train_loss = 0
        correct = 0
        total = 0
        early_stop_param = 0.01
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            if isinstance(optimizer, Sls):
                def closure():
                    output = model(data)
                    loss = criterion(output, target)
                    return loss
                optimizer.step(closure)
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            running_loss = loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch:d} [{batch_idx * len(data):d}/{len(train_loader.dataset):d} '
                    f'{100. * batch_idx / len(train_loader):.0f}] \tLoss: {loss.item():.6f} | '
                    f'Acc: {100. * correct / total:.3f}')

                if running_loss < early_stop_param:
                    print("Early Stopping !!!!!!!!!!")
                    break
                running_loss = 0.0

        if logger is not None:
            logger.write(dict(train_accuracy=100. * correct / total, loss=loss.item()), epoch)
        

    def test(self):
        pass