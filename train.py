import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn import metrics
from data_loader import data_loader
from Net import CGRE
from config import config
from eval import valid
from utils import AverageMeter, setup_seed, get_logger
from eval import test

def train(train_loader, test_loader, logger, opt):
    word_vec_dir = train_loader.dataset.word_vec_dir
    edge = train_loader.dataset.edge
    constraint = train_loader.dataset.constraint
    type_num = train_loader.dataset.type_num
    rel_num = train_loader.dataset.rel_num
    ckpt = os.path.join(opt['save_dir'], opt['name'] + '.pth.tar')
    model = CGRE(word_vec_dir, edge, constraint, type_num, rel_num, opt)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.rel_weight())
    optimizer = optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=1e-5)
    best_result = 0
    for epoch in range(opt['epoch']):
        logger.info('Epoch: %d' % epoch)
        model.train()
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for i, data in enumerate(train_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            word, pos1, pos2, ent1, ent2, mask, type, scope, rel = data
            output = model(word, pos1, pos2, ent1, ent2, mask, type, scope, rel)
            loss = criterion(output, rel)
            _, pred = torch.max(output, -1)
            acc = (pred == rel).sum().item() / rel.shape[0]
            pos_total = (rel != 0).sum().item()
            pos_correct = ((pred == rel) & (rel != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            # Optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info('[Train] loss: %f, acc: %f, pos_acc: %f' % (avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
        if (epoch + 1) % opt['val_iter'] == 0 and avg_pos_acc.avg > 0.6:
            y_true, y_pred = valid(test_loader, model)
            result = metrics.average_precision_score(y_true, y_pred)
            logger.info('[Test] AUC: %f' % result)
            if result > best_result:
                logger.info("Best result!")
                best_result = result
                torch.save({'state_dict': model.state_dict()}, ckpt)


if __name__ == '__main__':
    opt = vars(config())
    setup_seed(opt['seed'])
    logger = get_logger(opt['log_dir'], 'train_result.log')
    train_loader = data_loader(opt['train'], logger, opt, shuffle=True, training=True)
    test_loader = data_loader(opt['test'], logger, opt, shuffle=False, training=False)
    train(train_loader, test_loader, logger, opt)
    test(test_loader, None, logger, opt)
