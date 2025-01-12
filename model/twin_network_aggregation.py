import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
import os
import math
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import torchvision.models as models
from itertools import combinations
np.random.seed(0)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def load_npz_data(npz_folder, num_samples=10, times=5):
    train_data, test_data = [], []
    categories = os.listdir(npz_folder)

    for category in categories:
        fn = os.path.join(npz_folder, category)
        data = np.load(fn, allow_pickle=True)
        X = data[data.files[0]] 
        y = data[data.files[1]] 
        indices = np.random.permutation(len(X))
        train_indices = indices[:num_samples]
        test_num_samples = min(times * num_samples, len(X) - num_samples)
        test_indices = indices[num_samples:num_samples + test_num_samples]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        train_data.append((X_train, y_train))
        test_data.append((X_test, y_test))
        
    return train_data, test_data


class PcapDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32') / 255
        self.X = self.X.reshape([-1, 3, 16, 256])
        self.y = y.astype('int')
        self.x_data = torch.from_numpy(self.X)
        self.y_data = torch.from_numpy(self.y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y)


def extract_data_and_labels(dataset):
    data = []
    labels = []
    for x, y in dataset:
        data.append(x.numpy())
        labels.append(y.numpy())
    return data, labels


def create_three_channel_data(X, y):
    num_samples = len(X)
    three_channel_data = []
    for i in range(num_samples):
        class_idx = y[i]
        class_indices = np.where(y == class_idx)[0]
        if len(class_indices) > 1:
            prev_idx = np.roll(class_indices, -1)[class_indices == i]
            next_idx = np.roll(class_indices, 1)[class_indices == i]
            current_session = X[i].reshape(1, 16, 256)
            prev_session = X[prev_idx].reshape(1, 16, 256)
            next_session = X[next_idx].reshape(1, 16, 256)
            three_channel_sample = np.concatenate((current_session, prev_session, next_session), axis=0)
        else:
            three_channel_sample = np.tile(X[i].reshape(1, 16, 256), (3, 1, 1))
        three_channel_data.append(three_channel_sample)
    return np.array(three_channel_data)


def create_pairs(X, y):
    pairs, labels = [], []
    num_classes = len(np.unique(y))
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
    for i in unique_classes:
        for j, k in combinations(class_indices[i], 2):
            pairs += [[X[j], X[k]]]
            labels += [1]
    for i, j in combinations(unique_classes, 2):
        for idx1 in class_indices[i]:
            for idx2 in class_indices[j]:
                pairs += [[X[idx1], X[idx2]]]
                labels += [0]

    return np.array(pairs), np.array(labels)


class ResNetSiamese(nn.Module):
    def __init__(self, model_path=None):
        super(ResNetSiamese, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        if model_path:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.base_model = self.base_model.to(device)
            self.base_model.load_state_dict(state_dict)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 128)
    def forward_once(self, x):
        x = self.base_model(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def distance(self, output1, output2):
        return (output1 - output2).norm(p=2, dim=1)



def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = torch.pow(y_pred, 2)
    square_margin = torch.pow(torch.clamp(margin-y_pred, min=0.0), 2)
    loss_similar = y_true * square_pred
    loss_dissimilar = (1 - y_true) * square_margin
    return torch.mean(loss_similar + loss_dissimilar)


def calculate_class_means(model, X_train, y_train):
    class_means = []
    class_to_label = {}
    unique_classes = np.unique(y_train)
    for i, cls in enumerate(unique_classes):
        class_idx = np.where(y_train == cls)[0]
        class_features = []
        for idx in class_idx:
            img = torch.from_numpy(X_train[idx]).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                feature = model.forward_once(img.view(1, 3, 16, 256))
            class_features.append(feature)
        class_mean = torch.mean(torch.stack(class_features), dim=0)
        class_means.append(class_mean)
        class_to_label[i] = cls
    return class_means, class_to_label


def train_model(model, train_loader,test_loader,X_train,y_train, optimizer, scheduler, epochs, unknown_class):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (pair, target) in enumerate(train_loader):
            data1, data2 = pair[:, 0, :, :, :], pair[:, 1, :, :, :]
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            optimizer.zero_grad()
            output1, output2 = model(data1, data2)
            loss = contrastive_loss(target.float(), model.distance(output1, output2), margin=1.0)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
        scheduler.step()
        train_accuracy = evaluate_train_model(model, train_loader)
        print(f'Training Accuracy after Epoch {epoch+1}: {train_accuracy:.4f}')
        class_means, class_to_label = calculate_class_means(model,X_train , y_train)
        if((epoch+1) % 10 == 0):
            accuracy, similarities = evaluate_model(model, test_loader, X_train,y_train,class_means, class_to_label, unknown_class)
            print(f'Test Accuracy after Epoch {epoch+1}: {accuracy:.4f}')


def evaluate_train_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            input_data, labels = data
            input_data = input_data.to(device)
            labels = labels.to(device)
            output1, output2 = model(input_data[:, 0, :, :, :], input_data[:, 1, :, :, :])
            predicted = model.distance(output1, output2)
            predicted = (predicted.ravel() < 0.5).int()
            labels_binary = labels.int()

            total += labels.size(0)
            correct += torch.sum(predicted == labels_binary).item()
    accuracy = correct / total
    return accuracy


def calculate_dynamic_thresholds(class_means,class_to_label, X_train, y_train, model, k=2):
    thresholds = {}
    for i, class_mean in enumerate(class_means):
        class_indices = np.where(y_train == class_to_label[i])[0]
        distances = []
        for idx in class_indices:
            sample = torch.from_numpy(X_train[idx]).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                feature = model.forward_once(sample)
                distance = model.distance(feature, class_mean)
                distances.append(distance.item())
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        thresholds[i] = mean_distance + k * std_distance
    return thresholds


def calculate_max_distance_thresholds(class_means,class_to_label, X_train, y_train, model):
    thresholds = {}
    for i, class_mean in enumerate(class_means):
        class_indices = np.where(y_train == class_to_label[i])[0]
        max_distance = 0
        for idx in class_indices:
            sample = torch.from_numpy(X_train[idx]).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                feature = model.forward_once(sample)
                distance = model.distance(feature, class_mean)
                max_distance = max(max_distance, distance.item())
        thresholds[i] = max_distance
    return thresholds


from scipy.stats import norm
def calculate_gaussian_thresholds(class_means, class_to_label,X_train, y_train, model, confidence=0.95):
    thresholds = {}
    for i, class_mean in enumerate(class_means):
        class_indices = np.where(y_train == class_to_label[i])[0]
        distances = []
        for idx in class_indices:
            sample = torch.from_numpy(X_train[idx]).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                feature = model.forward_once(sample)
                distance = model.distance(feature, class_mean)
                distances.append(distance.item())
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        z_score = norm.ppf(confidence)
        thresholds[i] = mean_distance + z_score * std_distance
    return thresholds


def calculate_adaptive_thresholds_with_theory(class_means, class_to_label, X_train, y_train, model, k=2, confidence=0.95):
    thresholds = {}
    for i, class_mean in enumerate(class_means):
        class_indices = np.where(y_train == class_to_label[i])[0]
        distances = []
        for idx in class_indices:
            sample = torch.from_numpy(X_train[idx]).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                feature = model.forward_once(sample)
                distance = model.distance(feature, class_mean)
                distances.append(distance.item())
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = 3
        lower_bound = mean_distance - threshold * std_distance
        upper_bound = mean_distance + threshold * std_distance
        filtered_distances = [d for d in distances if lower_bound <= d <= upper_bound]
        mean_distance = np.mean(filtered_distances)
        std_distance = np.std(filtered_distances)
        z_score = norm.ppf((1 + confidence) / 2)
        adjusted_threshold = mean_distance + z_score * std_distance
        min_threshold = mean_distance + std_distance * np.log(len(class_indices))
        thresholds[i] = max(adjusted_threshold, min_threshold)
    
    return thresholds


def calculate_adaptive_thresholds_with_evt(class_means, class_to_label, X_train, y_train, model, k=2, confidence=0.95):
    thresholds = {}
    for i, class_mean in enumerate(class_means):
        class_indices = np.where(y_train == class_to_label[i])[0]
        distances = []
        for idx in class_indices:
            sample = torch.from_numpy(X_train[idx]).unsqueeze(0).float().to(device)
            model.eval()
            with torch.no_grad():
                feature = model.forward_once(sample)
                distance = model.distance(feature, class_mean)
                distances.append(distance.item())
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold_t = mean_distance + 2 * std_distance
        excesses = [d - threshold_t for d in distances if d > threshold_t]
        if len(excesses) < 2:
            print(f"Warning: Few data points above threshold for class {class_to_label[i]}. Consider adjusting the threshold.")
            continue
        gamma_hat, sigma_hat = estimate_gpd_parameters(excesses)
        q = 0.9
        n = len(distances)
        Np = len(excesses)
        T = threshold_t + (sigma_hat / gamma_hat) * (((q * n) / Np) ** (-gamma_hat) - 1)

        thresholds[i] = T

    return thresholds

def estimate_gpd_parameters(excesses):
    def log_likelihood(gamma, sigma, data):
        return -len(data) * np.log(sigma) - (1 + 1 / gamma) * np.sum(np.log(1 + gamma * (data / sigma)))
    from scipy.optimize import minimize
    result = minimize(lambda params: -log_likelihood(params[0], params[1], excesses), [0.5, 1], method='Nelder-Mead')
    gamma_hat, sigma_hat = result.x
    return gamma_hat, sigma_hat




def evaluate_model(model, test_loader, X_train,y_train,class_means, class_to_label, unknown_class):
    if(unknown_class is not None):
        n_classes = len(class_to_label) + unknown_class
    else:
        n_classes = len(class_to_label)
    class_incorrect_count = {i: [0] * n_classes for i in range(1, n_classes + 1)}
    correct_per_class = {i: 0 for i in range(1, n_classes + 1)}
    total_per_class = {i: 0 for i in range(1, n_classes + 1)}
    
    correct = 0
    total = 0
    if(unknown_class is not None):
        correct_unknown = 0
        total_unknown = 0
    similarities = []
    model.eval()
    thresholds_max = calculate_max_distance_thresholds(class_means,class_to_label, X_train, y_train, model)
    thresholds_dynamic = calculate_dynamic_thresholds(class_means,class_to_label, X_train, y_train, model)
    thresholds_gaussian = calculate_gaussian_thresholds(class_means,class_to_label, X_train, y_train, model)
    thresholds = calculate_adaptive_thresholds_with_theory(class_means,class_to_label, X_train, y_train, model)
    print(f'the class thresholds max distance is :{thresholds_max}')
    print(f'the class thresholds dynamic is :{thresholds_dynamic}')
    print(f'the class thresholds gaussian is :{thresholds_gaussian}')
    print(f'the class thresholds gaussian is :{thresholds}')
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)
            output = model.forward_once(test_data)
            batch_similarities = []
            for mean in class_means:
                mean = mean.to(device)
                similarity = model.distance(output, mean)
                batch_similarities.append(similarity)
            batch_similarities = torch.stack(batch_similarities, dim=1)
            predicted_indices = torch.argmin(batch_similarities, dim=1)
            predicted_labels = torch.tensor([class_to_label[int(idx.item())] for idx in predicted_indices], device=device)
            if (unknown_class is not None):
                for i, idx in enumerate(predicted_indices):
                    min_distance = batch_similarities[i, idx].item()
                    class_id = int(idx.item())
                    if(thresholds[class_id] < 0.1):
                        thresholds_jz = 0.1
                    else:
                        thresholds_jz = thresholds[class_id]
                    if min_distance > thresholds_jz:  
                        predicted_labels[i] = 999
                    
                correct_unknown += ((test_labels == 999) & (predicted_labels == 999)).sum().item()
                total_unknown += (test_labels == 999).sum().item()
            for true_label, predicted_label in zip(test_labels, predicted_labels):
                if(true_label.item() == 999):
                    index_x = n_classes
                else:
                    index_x = true_label.item()
                if(predicted_label.item() == 999):
                    index_y = n_classes
                else:
                    index_y = predicted_label.item()

                total_per_class[index_x] += 1
                if predicted_label.item() == true_label.item():
                    correct_per_class[index_x] += 1
                else:
                    class_incorrect_count[index_x][index_y-1] += 1
            correct += (predicted_labels == test_labels).sum().item()
            total += test_labels.size(0)

            similarities.extend(batch_similarities.cpu().numpy())
    for i in range(1, n_classes + 1):
        if(total_per_class[i] == 0):
            continue
        print(f'Class {i} count is: {total_per_class[i]}')
        accuracy = correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0
        print(f"Class {i} accuracy is: {accuracy:.4f}")
        for j in range(0, n_classes):
            if(i == j+1):
                print(f'Class {i}  classified as class {j+1}:{correct_per_class[i]}')
            if(class_incorrect_count[i][j] > 0):
                print(f"Class {i}  classified as class {j+1}: {class_incorrect_count[i][j]} times")
        
    if(unknown_class is None):
        accuracy = correct / total
        return accuracy, np.array(similarities)
    else:
        accuracy = correct / total
        if(total_unknown == 0):
            unknown_accuracy = 0
            print(f'there is no unknown samples!')
        else:
            unknown_accuracy = correct_unknown/total_unknown
        print(f'total unknown samples size : {total_unknown}')
        print(f'correct predicted unknown samples size : {correct_unknown}')
        print(f'unknown detect accuracy is : {unknown_accuracy}')
        return accuracy, np.array(similarities)

def load_and_preprocess_data(train_attack_X, train_attack_Y, test_attack_X, test_attack_Y, batch_size, test_batch_size):
    train_attack_X_three_channel = create_three_channel_data(train_attack_X, train_attack_Y)
    test_attack_X_three_channel = create_three_channel_data(test_attack_X, test_attack_Y)
    train_dataset = PcapDataset(train_attack_X_three_channel, train_attack_Y)
    test_dataset = PcapDataset(test_attack_X_three_channel, test_attack_Y)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)
    all_X = []
    all_y = []
    for i in range(len(train_dataset)):
        x, y = train_dataset[i]
        all_X.append(x.numpy())
        all_y.append(y.numpy())
    all_X = np.array(all_X)
    all_y = np.array(all_y)
    pairs_train, labels_train = create_pairs(all_X, all_y)
    print(f'pairs for train is {pairs_train.shape}')
    pairs_tensor = torch.tensor(pairs_train, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_train, dtype=torch.float32)
    train_pairs_dataset = TensorDataset(pairs_tensor, labels_tensor)
    train_pairs_loader = DataLoader(train_pairs_dataset, batch_size, shuffle=True)
    
    return train_pairs_loader, test_loader, train_dataset


def train_matching_network(model, train_pairs_loader, test_loader,X_train,y_train, \
                           optimizer, scheduler, epochs, unknown_class = None):
    model.to(device)
    train_model(model, train_pairs_loader, test_loader,X_train,y_train, \
                optimizer, scheduler, epochs, unknown_class)

