import os
import numpy as np
import glob
import argparse

from utils.metrics import *
from utils.loss import *
from utils.dataset import *

from net.inception import *

def parse_arg():
    model_nms = ["GoogLeNet", "PSGoogLeNet", "PCGoogLeNet", "PGoogLeNet"]
    data_nms = ["PETA", "RAP", "PA100K"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name in ' + str(model_nms) + '.')
    parser.add_argument('-s', '--save', type=str, default="",
                        help='The save name.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-d', '--data', type=str, default="",
                        help='The dataset need to be trained.')
    parser.add_argument('-w', '--weights', type=str, default="",
                        help='The weight file need to be loaded.')
    parser.add_argument('-c', '--classes', type=int, default=0,
                        help='The class number.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    if args.data == "" or args.data not in data_nms:
        raise RuntimeError('NO DATABASE FOUND IN ' + str(data_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------testing begining---------------------")
    args = parse_arg()
    model_prefix = "../models/" + args.data + "/" + args.model + "/"
    result_prefix = "../results/" + args.data + "/" + args.model + "/"
    os.makedirs(result_prefix, exist_ok=True)
    nb_class = args.classes
    save_name = args.save
    gpus = args.gpus.split(',')
    gpus = [int(x) for x in gpus if x != ""]
    
    ### Dataset & DataLoader
    if args.model == "GoogLeNet":
        indices = None
    else:
        if args.data == "PETA":
            whole = [1,2,3,4,5,16,34]
            hs = [0,8,20,21,25,28,36,37,44,54]
            ub = [15,19,23,27,30,39,40,46,50,51,55,56,58,59,60]
            lb = [10,14,18,22,24,29,31,32,45,47,53,57]
            sh = [9,26,42,43,48,49]
            at = [6,7,11,12,13,17,33,35,38,41,52]
            parts = [len(whole), len(hs), len(ub), len(lb), len(sh), len(at)]
        elif args.data == "RAP":
            whole = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]
            hs = [9,10,11,12,13,14]
            ub = [15,16,17,18,19,20,21,22,23]
            lb = [24,25,26,27,28,29]
            sh = [30,31,32,33,34]
            at = [35,36,37,38,39,40,41,42]
            parts = [len(whole), len(hs), len(ub), len(lb), len(sh), len(at)]
        indices = list(np.hstack((whole, hs, ub, lb, sh, at))[:nb_class])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = PARDataset(args.data, train=False, indices=indices, transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    ### Network
    if args.model == "GoogLeNet":
        arguments = nb_class
    else:
        arguments = parts
    net = globals()[args.model](classes = arguments)
    net.load_state_dict(torch.load(args.weight))
    if len(gpus) >= 1:
        device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
        """
        os.environment["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net, device_ids=gpus)
        net.to(device=gpus[0])
    print(net)
    
    ### Test
    #sigmoid = nn.Sigmoid()
    predictions_np = []
    labels_np = []
    for inputs, labels in test_dataloader:
        if len(gpus) >= 1:
            inputs = inputs.to(device=gpus[0])## 训练数据放在主设备
            labels = labels.to(device=gpus[0])
            labels = labels.float()
            
        outputs = net(inputs)
        #predictions = sigmoid(outputs)
        predictions = outputs
        test_loss += criterion(predictions, labels).item()
        if len(predictions_np) == 0:
            predictions_np = np.sign(predictions.cuda().data.cpu().numpy() - 0.5)
            labels_np = np.sign(labels.cuda().data.cpu().numpy() - 0.5)
        else:
            predictions_np = np.vstack((predictions_np, np.sign(predictions.cuda().data.cpu().numpy() - 0.5)))
            labels_np = np.vstack((labels_np, np.sign(labels.cuda().data.cpu().numpy() - 0.5)))
                
    np.save(predictions_np, result_prefix + save_name + "_" + w + "_predictions.npy")
    results = calculate_accuracy(labels_np, predictions_np)
    with open(result_prefix + save_name + "_" + w + "_results.file", "wb") as f:
        pickle.dump(results, f)
    print(result_prefix + save_name + "_" + w + '_results.file')