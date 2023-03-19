import torch
import statistics
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score

CLASSES_LIST = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea"]

def train_model(model, criterion, optimizer, model_path, device, epochs, train_loader, val_loader):
    '''
    Trains a PyTorch model on a given dataset.
    
    Args:
    - model (torch.nn.Module): The model you want to train.
    - criterion (torch.nn.Module): The loss criterion you want to use.
    - optimizer (torch.optim.Optimizer): The optimizer you want to use.
    - dataloader (torch.utils.data.DataLoader): The dataloader containing the training data.
    - model_path (str): The path you want to save your model to.
    - device (torch.device): Device to use for training.
    '''
    training_loss = []
    validation_loss = []

    total_CW_acc = []
    total_CW_prec = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
        accuracy = 100 * correct / total

        print(f"\n\nEpoch {epoch+1}\nTraining accuracy {accuracy:.4f} - Training loss: {running_loss/total:.4f} ")

        # Evaluate on validation set
        with torch.no_grad():
            print("validation set:")
            val_loss, val_acc, class_wise_acc, class_wise_precision = evaluate(model, criterion, val_loader, device, False)
            class_wise_acc = [tensor[0].item() for tensor in class_wise_acc]
            total_CW_acc.append(class_wise_acc)
            total_CW_prec.append(class_wise_precision)

        
        training_loss.append(running_loss/total)
        validation_loss.append(val_loss)


    mean_acc = []
    mean_prec = []
    for i in range(epochs):
        mean_acc.append(statistics.mean(total_CW_acc[i]))
        mean_prec.append(statistics.mean(total_CW_prec[i]))

    total_CW_acc = list(np.transpose(np.array(total_CW_acc)))
    total_CW_prec = list(np.transpose(np.array(total_CW_prec)))

    
    print(len(mean_acc), mean_acc)

    x = np.arange(epochs)
    legend = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea", "mean"]

    for i in range(len(CLASSES_LIST)):
        plt.plot(x, total_CW_acc[i], linestyle="--")
    plt.plot(x, mean_acc, linewidth=2, color='red')
    plt.legend(legend)
    plt.title("Class-wise accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("CW_accuracy.png")
    plt.figure()

    for i in range(len(CLASSES_LIST)):
        plt.plot(x, total_CW_prec[i], linestyle="--")
    plt.plot(x, mean_prec, linewidth=2, color='red')
    plt.legend(legend)
    plt.title("Class-wise precision")
    plt.xlabel("epochs")
    plt.ylabel("precision")
    plt.savefig("CW_precision.png")
    plt.figure()

    plt.plot(x, training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig("training_loss.png")
    plt.figure()

    plt.plot(x, validation_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation loss")
    plt.savefig("validation_loss.png")

    torch.save(model.state_dict(), model_path)


def evaluate(model, criterion, dataloader, device, per_class=True, show=False, test=False):
    '''
    Evaluates a dataset on a model. Is used for validation for validation set and test set.
    
    Args:
    - model (nn.Module): The model you want to evaluate
    - criterion (nn.Module): The criterion you want to use
    - dataloader (DataLoader): The dataloader object containing the data you want to evaluate
    - device (torch.device): The device you want to run the evaluation on
    - per_class (bool): If True, it will print the accuracy for each class individually
    - show (bool): If True, it will print one image from each batch with the prediction and label. Default False

    Returns:
    - float: Loss of the model
    - float: Accuracy of the model
    '''
    
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    class_wise_precision = [0,0,0,0,0,0]

    class_wise_acc = [[],[],[],[],[],[]]

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)          

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #Calculate precision
            class_wise_precision += precision_score(labels.cpu(), predicted.cpu(), average=None)
            #Calculating class-wise accuracy
            mask1 = torch.eq(labels, predicted)
            for i in range(len(CLASSES_LIST)):
                total_counts[i] += (labels == i).sum().item()
                
                #Find the accuracy per class
                mask2 = torch.eq(labels, i) 
                mask3 = torch.eq(predicted, i)
                mask4 = torch.logical_and(torch.logical_and(mask1, mask2), mask3)

                correct_counts[i] += torch.sum(mask4 == True,dim=0)
                
            if test:
                for i in range(3):
                    get_top_bottom_10(outputs, inputs, labels, i)
                plt.show()

                torch.save(outputs, 'test_softmax.pt')
                torch.save(labels, 'test_labels.pt')

           
            if show:
                image = np.transpose(inputs[0], (2, 1, 0))
                plt.imshow(image)
                plt.title(f"correct {CLASSES_LIST[labels[0]]}, predicted {CLASSES_LIST[predicted[0]]}")
                plt.figure()

        average_precision = sum(class_wise_precision) / len(class_wise_precision)
        
        print(f"Average accuracy is {100 * correct / total:.4f}, loss is {running_loss/total}")
        print(f"class-wise accuracy is:", end=" ")
        for i in range(len(CLASSES_LIST)):
            accuracy = 100 * correct_counts[i] / total_counts[i]
            class_wise_acc[i].append(accuracy)
            print("{} : {:.2f}%".format(CLASSES_LIST[i], accuracy), end=" - ")

        print(f"\nAverage precision is {average_precision:.4f}")
        print(f"class-wise precision is:", end=" ")
        for i in range(len(CLASSES_LIST)):
            print("{} : {:.2f}".format(CLASSES_LIST[i], class_wise_precision[i]), end=" - ")



    val_loss = running_loss / total

    accuracy = 100 * correct / total

    return val_loss, accuracy, class_wise_acc, class_wise_precision


def get_top_bottom_10(softmax, inputs, labels, class_nr):
    """Find the top-10 and bottom-10 images for a given class based on the softmax values, and display them in a plot.

    Args:
        softmax (torch.Tensor): A tensor of the softmax values for the six classes for each image.
        inputs (torch.Tensor): The input image as a tensor.
        labels (torch.Tensor): A tensor of the correct label for each image.
        class_nr (int): The number between 0 and 5, corresponding to the class we want to find the top-10 and bottom-10 images for.
    """

    _, preds = torch.max(softmax, dim=1)

    class_indices = (labels == class_nr).nonzero().squeeze()
    class_softmax = softmax[class_indices, class_nr]
    
    # Find the top-10 and bottom-10 images based on the softmax values
    top_indices = torch.argsort(class_softmax, descending=True)[:10]
    bottom_indices = torch.argsort(class_softmax, descending=False)[:10]
    
    fig, axs = plt.subplots(2, 10, figsize=(20, 4))
    
    # Display the top-10 images
    for i, idx in enumerate(top_indices):
        img = inputs[class_indices[idx]].permute(1, 2, 0).cpu().numpy()
        axs[0, i].imshow(img)
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Pred: {CLASSES_LIST[preds[class_indices[idx]].item()]}")

    # Display the bottom-10 images
    for i, idx in enumerate(bottom_indices):
        img = inputs[class_indices[idx]].permute(1, 2, 0).cpu().numpy()
        axs[1, i].imshow(img)
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Pred: {CLASSES_LIST[preds[class_indices[idx]].item()]}")
