import os
import json
import torch
# from symb_xai.model.gcn import GCN
from symb_xai.model.gin import GIN
import time

models_repository = '/Users/thomasschnake/Research/Projects/symbolic_xai/saved_models/mutagenicity'

def train_model(model, dataset, train_idx, test_idx, learning_rate, num_epochs, dataset_name, architecture, gcn_layers, mlp_layers, hidden_dim, test_epochs=10):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time_tag = int(time.time())
    best_loss = float('inf')
    # Train the model

    for epoch in range(num_epochs):
        for idx in train_idx:
            trainX = dataset[idx].x
            trainEdgeIndex = dataset[idx].edge_index
            trainY = dataset[idx].y
            outputs = model(trainX, trainEdgeIndex)
            optimizer.zero_grad()
            
            # obtain the loss function
            loss = criterion(outputs.unsqueeze(0), torch.eye(2)[trainY])
            loss.backward()
            
            optimizer.step()
        if (epoch+1) % test_epochs == 0:
            test_acc = 0
            for idx in test_idx:
                testX = dataset[idx].x
                testEdgeIndex = dataset[idx].edge_index
                testY = dataset[idx].y
                outputs = model(testX, testEdgeIndex)
                test_acc += (torch.argmax(outputs) == testY).float()
            test_acc /= len(test_idx)
            print("Epoch: %d, loss: %1.5f, test_acc: %1.5f" % (epoch+1, loss.item(), test_acc))
            if loss.item() < best_loss: 
                best_loss = loss.item()
                filename = f'{dataset_name}_{architecture}_{start_time_tag}'
                torch.save(model.state_dict(), f'{models_repository}/{filename}.pth')
                model_info =   {'dataset': dataset_name,
                                'architecture': architecture,
                                'learning rate': learning_rate, 
                                'epoch': epoch, 
                                'total_epochs': num_epochs,
                                'criterion': str(criterion),
                                'optimizer': str(optimizer),
                                'input_features': dataset.num_features,
                                'num_classes': dataset.num_classes,
                                'gcn_layers': gcn_layers,
                                'mlp_layers': mlp_layers,
                                'hidden_dim': hidden_dim,
                                'bias': model.bias,
                                'test_acc': float(test_acc)
                                }
                json.dump(model_info, open(f'{models_repository}/{filename}.json', 'w'))
    print(f'Finished Training model {filename}.pt with test accuracy {float(test_acc):.4f}')

def load_best_model(dataset_name):
    best_acc = 0
    for file in os.listdir(f'{models_repository}/'):
        if file[:len(dataset_name)] == dataset_name:
            filename = file.split('.')[0]
            model_info = json.load(open(f'{models_repository}/{filename}.json', 'r'))
            if model_info['test_acc'] > best_acc:
                best_acc = model_info['test_acc']
                best_model_filename = filename
    model_info = json.load(open(f'{models_repository}/{best_model_filename}.json', 'r'))
    hidden_dim = model_info['hidden_dim']
    gcn_layers = model_info['gcn_layers']
    mlp_layers = model_info['mlp_layers']
    architecture = model_info['architecture']
    input_features = model_info['input_features']
    num_classes = model_info['num_classes']
    bias = model_info['bias'] if 'bias' in model_info else True
    if architecture == 'GIN':
        model = GIN(hidden_dim=hidden_dim, input_dim=input_features, gcn_layers=gcn_layers, mlp_layers=mlp_layers, nbclasses=num_classes, node_level=False, directed=False, regression=False, bias=bias)
    elif architecture == 'GCN':
        model = GCN(hidden_dim=hidden_dim, input_dim=input_features, gcn_layers=gcn_layers, nbclasses=num_classes, node_level=False, directed=False, regression=False)
    else:
        raise NotImplementedError(f'Architecture {architecture} not implemented')
    model.load_state_dict(torch.load(f'{models_repository}/{best_model_filename}.pth'))
    print(f'Loaded model {best_model_filename}.pth with test accuracy {best_acc:.4f}')
    return model

def remove_bad_models(dataset_name):
    best_acc = 0
    for file in os.listdir(f'{models_repository}/'):
        if file[:len(dataset_name)] == dataset_name and file.split('.')[1] == 'pth' and not os.path.exists(f'{models_repository}/{file.split(".")[0]}.json'):
            os.remove(f'{models_repository}/{file}')
            print(f'Removed model {file} without json file')

    for file in os.listdir(f'{models_repository}/'):
        if file[:len(dataset_name)] == dataset_name and file.split('.')[1] == 'json':
            filename = file.split('.')[0]
            model_info = json.load(open(f'{models_repository}/{filename}.json', 'r'))
            if model_info['test_acc'] > best_acc:
                best_acc = model_info['test_acc']

    for file in os.listdir(f'{models_repository}/'):
        if file[:len(dataset_name)] == dataset_name and file.split('.')[1] == 'json':
            filename = file.split('.')[0]
            model_info = json.load(open(f'{models_repository}/{filename}.json', 'r'))
            if model_info['test_acc'] < best_acc:
                os.remove(f'{models_repository}/{filename}.json') if os.path.exists(f'{models_repository}/{filename}.json') else None
                os.remove(f'{models_repository}/{filename}.pth') if os.path.exists(f'{models_repository}/{filename}.pth') else None
                print(f'Removed model {filename}.pt with test accuracy {model_info["test_acc"]:.4f}')
    
def evaluate_model(model, dataset):
    pred = []
    true = []
    for idx in range(len(dataset)):
        testX = dataset[idx].x
        testEdgeIndex = dataset[idx].edge_index
        testY = dataset[idx].y
        outputs = model(testX, testEdgeIndex)
        pred.append(torch.argmax(outputs).item())
        true.append(testY.item())
        
    acc = sum([1 if p == t else 0 for p, t in zip(pred, true)]) / len(pred)
    acc_pos = sum([1 if p == t else 0 for p, t in zip(pred, true) if t == 1]) / sum([1 if t == 1 else 0 for t in true])
    acc_neg = sum([1 if p == t else 0 for p, t in zip(pred, true) if t == 0]) / sum([1 if t == 0 else 0 for t in true])
    balanced_acc = (acc_pos + acc_neg) / 2
    return acc, balanced_acc, acc_pos, acc_neg    