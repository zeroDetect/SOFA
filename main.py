from VAE_model_aggregation import *
from twin_network_aggregation import *
import random
import numpy as np
import pickle
NEW_CLASS_TEST = 0

def save_data_to_npz(data, labels, filename):
    np.savez(filename, X=data, Y=labels)


def save_model(model, threshold, category, filename):
    model_data = {
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'category': category
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

def main():
    parameters = {
        'hidden_dim': 128,
        'latent_dim': 2,
        'batch_size': 64,
        'test_size': 0.2,
        'epochs': 100,
        'learning_rate':1e-3,
        'fsl_train_number': 15,
        'fsl_batch_size':5,
        'fsl_test_batch_size':10,
        'fsl_epochs':2,
        'unknown_class':None
    }
    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_filedir = 'data/benign'
    detect_filedir = 'data/attack'
    detect_files = find_npz_files(detect_filedir)
    all_attack_data, all_attack_label = read_npz_files(detect_files)
    all_attack_X = np.concatenate(all_attack_data, axis=0)
    all_attack_Y = np.concatenate(all_attack_label, axis=0)
    if(parameters['unknown_class'] is not None):
        attack_train_X, attack_train_Y, attack_test_X, attack_test_Y \
        = split_unknown_by_class(all_attack_X, all_attack_Y, num_samples_per_class=parameters['fsl_train_number'], \
                                 unknown_class=parameters['unknown_class'])
    else:
        attack_train_X, attack_train_Y, attack_test_X, attack_test_Y \
            = split_data_by_class(all_attack_X, all_attack_Y, split_type='num', num_samples_per_class=parameters['fsl_train_number'])
    save_data_to_npz(attack_train_X, attack_train_Y, 'data/attack_split/attack_train.npz')
    save_data_to_npz(attack_test_X, attack_test_Y, 'data/attack_split/attack_test.npz')
    train_files = find_npz_files(train_filedir)
    if(NEW_CLASS_TEST == 1):
        print("new class comes out===============")
        if len(train_files) < 2:
            print("The number of files is less than two and cannot be extracted")
        else:
            new_files = random.sample(train_files, 2)
            for file in new_files:
                train_files.remove(file)
        new_benign_data, new_benign_label = read_npz_files(new_files)
        new_benign_X = np.concatenate(new_benign_data, axis=0)
        new_benign_Y = np.concatenate(new_benign_label, axis=0)
        unique_labels, label_counts = np.unique(new_benign_Y[:,0], return_counts=True)
        labels_to_remove = unique_labels[label_counts < 100]
        labels_to_keep = unique_labels[label_counts >= 100]
        for label, count in zip(unique_labels, label_counts):
            if(count > 100):
                print(f"The number of samples of category {label} is: {count}, which meets the training conditions")
            else:
                print(f"The number of samples of category {label} is: {count}, which does not meet the training conditions and will continue to wait for samples")
        mask = np.isin(new_benign_Y[:,0], labels_to_keep)
        new_benign_Y = new_benign_Y[mask] 
        new_benign_X = [new_benign_X[i] for i in range(len(new_benign_X)) if mask[i]]
        new_benign_train_X, new_benign_train_Y, new_benign_test_X, new_benign_test_Y \
            = split_data_by_class(new_benign_X, new_benign_Y, split_type='ratio', split_value=0.8,num_samples_per_class=None)
        new_benign_train_Y[:, 1] = np.where(new_benign_train_Y[:, 1] == '1', 0, new_benign_train_Y[:, 1])
        new_benign_test_Y[:, 1] = np.where(new_benign_test_Y[:, 1] == '1', 0, new_benign_test_Y[:, 1])
        new_train_benign_Y_df = pd.DataFrame(new_benign_train_Y, columns=['ip_port_proto', 'label'])
        new_test_benign_Y_df = pd.DataFrame(new_benign_test_Y, columns=['ip_port_proto', 'label'])
        new_categories = new_train_benign_Y_df['ip_port_proto'].unique()
        new_test_data = {}
        for category in new_categories:
            new_train_category_indices = new_train_benign_Y_df[new_train_benign_Y_df['ip_port_proto'] == category].index
            print(f'now train for new category {category}=================')
            new_test_category_indices = new_test_benign_Y_df[new_test_benign_Y_df['ip_port_proto'] == category].index
            new_VAE_train_X = new_benign_train_X[new_train_category_indices]
            new_VAE_train_Y = new_benign_train_Y[new_train_category_indices]
            new_VAE_test_X = new_benign_test_X[new_test_category_indices]
            new_VAE_test_Y = new_benign_test_Y[new_test_category_indices]
            new_normalized_VAE_train_X, new_normalized_VAE_test_X = normalize_data(new_VAE_train_X, new_VAE_test_X)
            print(f'start training for new category {category}========')
            new_model, new_threshold = train_process(new_normalized_VAE_train_X, new_VAE_train_Y, parameters)
            save_model(new_model, new_threshold, category, f'model/model_{category}.pth')
            new_combined_VAE_test_X_slice = torch.tensor(new_normalized_VAE_test_X[:,16,4:110], dtype=torch.float32)
            new_combined_VAE_test_Y_numeric = np.array(new_VAE_test_Y[:, 1], dtype=float)
            new_combined_VAE_test_Y_numeric = np.where(new_combined_VAE_test_Y_numeric > 1, 1.0, new_combined_VAE_test_Y_numeric)
            new_combined_VAE_test_Y_slice = torch.tensor(new_combined_VAE_test_Y_numeric, dtype=torch.float32)
            new_test_dataset = TensorDataset(new_combined_VAE_test_X_slice, new_combined_VAE_test_Y_slice)
            new_test_loader = DataLoader(new_test_dataset, batch_size=parameters['batch_size'], shuffle=False)
            print(f'Testing for new VAE model {category} ========')
            results_df = test_process(new_model, new_threshold, new_test_loader, device)
    all_benign_data, all_benign_label = read_npz_files(train_files)
    all_benign_X = np.concatenate(all_benign_data, axis=0)
    all_benign_Y = np.concatenate(all_benign_label, axis=0)
    benign_train_X, benign_train_Y, benign_test_X, benign_test_Y \
        = split_data_by_class(all_benign_X, all_benign_Y, split_type='ratio', split_value=0.8,num_samples_per_class=None)
    benign_train_Y[:, 1] = np.where(benign_train_Y[:, 1] == '1', 0, benign_train_Y[:, 1])
    benign_test_Y[:, 1] = np.where(benign_test_Y[:, 1] == '1', 0, benign_test_Y[:, 1])
    save_data_to_npz(benign_train_X, benign_train_Y, 'data/benign_split/benign_train.npz')
    save_data_to_npz(benign_test_X, benign_test_Y, 'data/benign_split/benign_test.npz')

    train_benign_file = np.load('data/benign_split/benign_train.npz')
    test_benign_file = np.load('data/benign_split/benign_test.npz')
    train_attack_file = np.load('data/attack_split/attack_train.npz')
    test_attack_file = np.load('data/attack_split/attack_test.npz')
    train_benign_X = train_benign_file['X']#
    train_benign_Y = train_benign_file['Y']#

    test_benign_X = test_benign_file['X']#
    test_benign_Y = test_benign_file['Y']#

    train_attack_X = train_attack_file['X']#
    train_attack_Y = train_attack_file['Y']#

    test_attack_X = test_attack_file['X']#
    test_attack_Y = test_attack_file['Y']#
    train_benign_Y_df = pd.DataFrame(train_benign_Y, columns=['ip_port_proto', 'label'])
    test_benign_Y_df = pd.DataFrame(test_benign_Y, columns=['ip_port_proto', 'label'])
    categories = train_benign_Y_df['ip_port_proto'].unique()
    test_data = {}
    encoder_outputs = {} 
    for category in categories:
        train_category_indices = train_benign_Y_df[train_benign_Y_df['ip_port_proto'] == category].index
        print(f'now train for {category}=================')
        test_category_indices = test_benign_Y_df[test_benign_Y_df['ip_port_proto'] == category].index
        VAE_train_X = train_benign_X[train_category_indices]
        VAE_train_Y = train_benign_Y[train_category_indices]
        VAE_test_X = test_benign_X[test_category_indices]
        VAE_test_Y = test_benign_Y[test_category_indices]
        normalized_VAE_train_X, normalized_VAE_test_X = normalize_data(VAE_train_X, VAE_test_X)
        test_data[category] = {
            'normalized_VAE_test_X': normalized_VAE_test_X,
            'VAE_test_Y': VAE_test_Y
        }
        model, threshold = train_process(normalized_VAE_train_X, VAE_train_Y, parameters)
        save_model(model, threshold, category, f'model/model_{category}.pth')
        model.eval()
        encoder_output = model.encode(torch.tensor(normalized_VAE_train_X[:,16,4:110], dtype=torch.float32).to(device))
        latent_codes = encoder_output[:, :model.latent_dim]
        encoder_outputs[category] = latent_codes.detach().cpu().numpy() 
    with open(os.path.join('.', 'plot/encoder_outputs.pkl'), 'wb') as f:
        pickle.dump(encoder_outputs, f)
    with open(os.path.join('data', 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    train_pairs_loader, test_loader, train_dataset = \
        load_and_preprocess_data(train_attack_X[:, :16, :], train_attack_Y[:,1], test_attack_X[:, :16, :], test_attack_Y[:,1],\
                                  parameters['fsl_batch_size'], parameters['fsl_test_batch_size'])
    model_path = 'resnet18-f37072fd.pth'
    model_fsl = ResNetSiamese(model_path).to(device)
    optimizer_fsl = optim.Adam(model_fsl.parameters(), lr=0.0001)
    scheduler_fsl = lr_scheduler.StepLR(optimizer_fsl, step_size=30, gamma=0.1)
    X_train, y_train = extract_data_and_labels(train_dataset)
    train_matching_network(model_fsl, train_pairs_loader, test_loader,X_train,y_train, \
                           optimizer_fsl, scheduler_fsl, parameters['fsl_epochs'],\
                            parameters['unknown_class'])
    torch.save(model_fsl.state_dict(), 'model/resnet_siamese_model.pth')
    loaded_models_info  = {}
    model_dir = 'model'
    for filename in os.listdir(model_dir):
        if filename.startswith('model_') and filename.endswith('.pth'):
            filepath = os.path.join(model_dir, filename)
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            model_state_dict = model_data['model_state_dict']
            threshold = model_data['threshold']
            category = model_data['category']
            model = VAE(normalized_VAE_train_X[:, 16, 4:110].shape[1], \
                        parameters['hidden_dim'], parameters['latent_dim'])
            model.load_state_dict(model_state_dict)
            loaded_models_info[filename.replace(".pth", "")] = {
                'model': model,
                'threshold': threshold,
                'category': category
            }
    
    pkl_file_path = 'data/test_data.pkl'
    with open(pkl_file_path, 'rb') as f:
        test_data = pickle.load(f)
    all_results_df = {}
    for category, data in test_data.items():
        normalized_VAE_test_X = data['normalized_VAE_test_X']
        VAE_test_Y = data['VAE_test_Y']
        matched_attack_X, matched_attack_Y = match_and_select_data(category, test_attack_X, test_attack_Y)
        combined_VAE_test_X = np.concatenate((normalized_VAE_test_X, matched_attack_X), axis=0)
        combined_VAE_test_Y = np.concatenate((VAE_test_Y, matched_attack_Y), axis=0)
        combined_VAE_test_X_slice = torch.tensor(combined_VAE_test_X[:,16,4:110], dtype=torch.float32)
        
        combined_VAE_test_Y_numeric = np.array(combined_VAE_test_Y[:, 1], dtype=float)
        combined_VAE_test_Y_numeric = np.where(combined_VAE_test_Y_numeric > 1, 1.0, combined_VAE_test_Y_numeric)
        
        combined_VAE_test_Y_slice = torch.tensor(combined_VAE_test_Y_numeric, dtype=torch.float32)
        test_dataset = TensorDataset(combined_VAE_test_X_slice, combined_VAE_test_Y_slice)
        test_loader = DataLoader(test_dataset, batch_size=parameters['batch_size'], shuffle=False)
        print(f'Testing for VAE model {category} ========')
        results_df = test_process(loaded_models_info['model_'+ category]['model'], loaded_models_info['model_'+ category]['threshold'], test_loader, device)
        predictions = results_df['Predictions'].values
        combined_VAE_test_Y_extended = np.hstack((combined_VAE_test_Y, predictions[:, None]))
        all_results_df[category] = {
            'combined_VAE_test_Y_df': combined_VAE_test_Y_extended,
            'combined_VAE_test_X': combined_VAE_test_X
        }
    fsl_model_load = ResNetSiamese()
    fsl_model_load.load_state_dict(torch.load('model/resnet_siamese_model.pth', map_location=device))
    fsl_model_load.to(device)

    class_means, class_to_label = calculate_class_means(fsl_model_load,X_train , y_train)
    all_combined_VAE_test_X = []
    all_combined_VAE_test_Y_df = []

    for category, results in all_results_df.items():
        combined_VAE_test_Y_df = results['combined_VAE_test_Y_df']
        combined_VAE_test_X = results['combined_VAE_test_X']
        all_combined_VAE_test_X.append(combined_VAE_test_X)
        all_combined_VAE_test_Y_df.append(combined_VAE_test_Y_df)
    all_combined_VAE_test_X = np.concatenate(all_combined_VAE_test_X, axis=0)
    all_combined_VAE_test_Y_df = np.concatenate(all_combined_VAE_test_Y_df, axis=0)

    condition = (np.array(all_combined_VAE_test_Y_df[:, 2], dtype=float) == 1) \
        & (np.array(all_combined_VAE_test_Y_df[:, 1], dtype=float) >= 1)
    indices = np.where(condition)[0]
    

    condition2 = ((np.array(all_combined_VAE_test_Y_df[:, 1], dtype=float) >= 1) &\
    (np.array(all_combined_VAE_test_Y_df[:, 2], dtype=float) == 0))|\
    ((np.array(all_combined_VAE_test_Y_df[:, 1], dtype=float) == 0) &\
    (np.array(all_combined_VAE_test_Y_df[:, 2], dtype=float) >= 1))

    num_VAE_error = np.sum(condition2)
    total_length = len(all_combined_VAE_test_Y_df)
    ratio = num_VAE_error /total_length
    print(f'Total VAE test accuracy is {1-ratio}')

    selected_VAE_test_X = all_combined_VAE_test_X[indices]
    selected_VAE_test_Y = all_combined_VAE_test_Y_df[indices, 1]

    X_test_three_channel = create_three_channel_data(selected_VAE_test_X[:,:16,:], selected_VAE_test_Y)

    test_dataset = PcapDataset(X_test_three_channel, selected_VAE_test_Y)
    test_loader = DataLoader(test_dataset, batch_size=parameters['fsl_test_batch_size'], shuffle=False)

    if(len(indices) == 0):
        print(f'no attack type matching any benign tuple!')
    else:
        accuracy, similarities = evaluate_model(fsl_model_load, test_loader, X_train,y_train,class_means, class_to_label, parameters['unknown_class'])
        print(f'Total FSL Test Accuracy : {accuracy:.4f}')

        total_FSL_num = np.sum(condition)
        total_accuracy = (total_FSL_num*accuracy + (total_length - num_VAE_error))/(total_FSL_num + total_length)
        print(f'Total Accuracy : { total_accuracy }')


def split_unknown_by_class(all_X, all_Y, num_samples_per_class=None, unknown_class = 2):
    classes = np.unique(all_Y[:, 0])
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    current_label = 1
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)
    unknown_classes = np.random.choice(classes, unknown_class, replace=False)
    print(f'Unknown classes: {unknown_classes}')
    for cls in unknown_classes:
        class_mask = all_Y[:, 0] == cls
        class_X = all_X[class_mask]
        class_Y = all_Y[class_mask]
        test_X.append(class_X)
        class_Y[:, 1] = 999
        test_Y.append(class_Y)

    for cls in classes:
        if cls in unknown_classes:
            continue
        class_mask = all_Y[:, 0] == cls
        class_X = all_X[class_mask]
        class_Y = all_Y[class_mask]
        print(f'class vs labels relation is {cls}--{current_label}')
        print(f'the class {cls} shape is {class_X.shape}')
        indices = np.arange(class_X.shape[0])
        np.random.shuffle(indices)
        class_X = class_X[indices]
        class_Y = class_Y[indices]

        if num_samples_per_class is not None:
            split_point = num_samples_per_class
        else:
            split_point = int(class_X.shape[0] * 0.8) 
        train_X.append(class_X[:split_point])
        test_X.append(class_X[split_point:])
        if cls == 'benign' or '_6' in cls or '_17' in cls:
            train_Y.append(class_Y[:split_point, :])
            test_Y.append(class_Y[split_point:, :])
        else:
            new_labels = np.full((class_Y[:split_point].shape[0], 1), current_label, dtype=int)
            train_Y.append(np.hstack((class_Y[:split_point, :1], new_labels))) 
            new_labels_test = np.full((class_Y[split_point:].shape[0], 1), current_label, dtype=int)
            test_Y.append(np.hstack((class_Y[split_point:, :1], new_labels_test)))
            current_label += 1

    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)
    test_X = np.concatenate(test_X, axis=0)
    test_Y = np.concatenate(test_Y, axis=0)

    return train_X, train_Y, test_X, test_Y


def split_data_by_class(all_X, all_Y, split_type='ratio', split_value=0.8, num_samples_per_class=None):
    classes = np.unique(all_Y[:, 0])
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    current_label = 1
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)
    for cls in classes:
        class_mask = all_Y[:, 0] == cls
        class_X = all_X[class_mask]
        class_Y = all_Y[class_mask]
        print(f'class vs labels relation is {cls}--{current_label}')
        print(f'the class {cls} shape is {class_X.shape}')
        indices = np.arange(class_X.shape[0])
        np.random.shuffle(indices)
        class_X = class_X[indices]
        class_Y = class_Y[indices]

        if split_type == 'ratio':
            split_point = int(split_value * class_X.shape[0])
        elif split_type == 'num':
            split_point = num_samples_per_class
        else:
            raise ValueError("split_type must be 'ratio' or 'num'")
        train_X.append(class_X[:split_point])
        test_X.append(class_X[split_point:])
        if cls == 'benign' or '_6' in cls or '_17' in cls:
            train_Y.append(class_Y[:split_point, :])
            test_Y.append(class_Y[split_point:, :])
        else:
            new_labels = np.full((class_Y[:split_point].shape[0], 1), current_label, dtype=int)
            train_Y.append(np.hstack((class_Y[:split_point, :1], new_labels))) 
            new_labels_test = np.full((class_Y[split_point:].shape[0], 1), current_label, dtype=int)
            test_Y.append(np.hstack((class_Y[split_point:, :1], new_labels_test)))
            current_label += 1
    train_X = np.concatenate(train_X, axis=0)
    train_Y = np.concatenate(train_Y, axis=0)
    test_X = np.concatenate(test_X, axis=0)
    test_Y = np.concatenate(test_Y, axis=0)

    return train_X, train_Y, test_X, test_Y



if __name__ == '__main__':
    main()
