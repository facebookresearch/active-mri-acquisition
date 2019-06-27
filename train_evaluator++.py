from util.rl.evaluator_plus_plus import EvaluatorDataset, EvaluatorPlusPlus, train

import torch
import torch.utils.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    dataset = EvaluatorDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    model = EvaluatorPlusPlus()
    train(model, data_loader, device)
