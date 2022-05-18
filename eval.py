import torch
import os
import sys
import numpy as np
from sklearn import metrics
from data_loader import data_loader
from Net import CGRE
from utils import AverageMeter, setup_seed, get_logger, config
import matplotlib.pyplot as plt


def test(loader, pn_loaders, logger, opt):
    word_vec_dir = loader.dataset.word_vec_dir
    edge = loader.dataset.edge
    constraint = loader.dataset.constraint
    type_num = loader.dataset.type_num
    rel_num = loader.dataset.rel_num
    ckpt = os.path.join(opt['save_dir'], opt['name'] + '.pth.tar')
    model = CGRE(word_vec_dir, edge, constraint, type_num, rel_num, opt)
    if torch.cuda.is_available():
        model = model.cuda()
    state_dict = torch.load(ckpt)['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            logger.warning(name + " not in this model!")
            continue
        own_state[name].copy_(param)
    torch.save({'state_dict': own_state}, ckpt)
    logger.info("=== Denoising Evaluation ===")
    y_true, y_pred = valid(loader, model)
    auc = compute_curve(y_true, y_pred, opt['log_dir'], opt['name'])
    logger.info("\rAUC: {0}".format(auc*100))
    if pn_loaders is not None:
        for i, pn_loader in enumerate(pn_loaders):
            if i == 0:
                mode = 'one'
            elif i == 1:
                mode = 'two'
            else:
                mode = 'all'
            logger.info("pn_mode: %s" % mode.capitalize())
            y_true_pn, y_pred_pn = valid(pn_loader, model)
            compute_pn(y_true_pn, y_pred_pn, logger)
    logger.info("=== Long-Tailed Relation Evaluation ===")
    if 'NYT' in opt['dataset']:
        rel_100, rel_200 = loader.dataset.rel_100, loader.dataset.rel_200
        hits_at_k(y_true, y_pred, rel_100, rel_200, rel_num, logger)


def valid(loader, model):
    model.eval()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if torch.cuda.is_available():
                data = [x.cuda() if torch.is_tensor(x) else x for x in data]
            rel = data[-1]
            output = model(data[:-1])
            label = rel.argmax(-1)
            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f'%(i+1, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    return y_true, y_pred


def compute_curve(y_true, y_pred, plot_dir, name):
    auc = metrics.average_precision_score(y_true, y_pred)
    order = np.argsort(-y_pred)
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)
    # auc = metrics.auc(recall, precision)
    save_dir = 'Curves/' + name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(save_dir + '/precision.npy', precision)
    np.save(save_dir + '/recall.npy', recall)
    draw(precision, recall, plot_dir, name)
    return auc


def draw(precision, recall, plot_dir, name):
    plt.plot(recall, precision, color='red', label=name, marker='o', lw=1, markevery=0.1, ms=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, name+'.png'), format="png")


def compute_pn(y_true, y_pred, logger):
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean()*100
    p200 = (y_true[order[:200]]).mean()*100
    p300 = (y_true[order[:300]]).mean()*100
    mean = (p100+p200+p300)/3
    logger.info("P@100: {0:.1f}, P@200: {1:.1f}, P@300: {2:.1f}, Mean: {3:.1f}".format(p100, p200, p300, mean))


def hits_at_k(y_true, y_pred, rel_100, rel_200, rel_num, logger):
    y_true = y_true.reshape(-1, rel_num - 1)
    y_pred = y_pred.reshape(-1, rel_num - 1)
    rel_100_num = 0
    rel_200_num = 0
    # micro
    k100 = np.zeros(4)  # [hits at 5, 10, 15, 20]
    k200 = np.zeros(4)
    # macro
    hash_100 = {}
    hash_200 = {}
    k100_rel = np.zeros((5, len(rel_100)))  # [hit@5, hit@10, hit@15, hit@20, rel_total_num]
    k200_rel = np.zeros((5, len(rel_200)))
    for a, (ins_true, ins_pred) in enumerate(zip(y_true, y_pred)):
        true_rel = np.where(ins_true == 1)[0]
        if len(true_rel) == 0:
            continue
        r = true_rel[0]
        if r in rel_100 or r in rel_200:
            k = np.sum(ins_pred > ins_pred[r])
            if r in rel_100:
                rel_100_num += 1
                if r not in hash_100:
                    hash_100[r] = len(hash_100)
                ind = hash_100[r]
                k100_rel[-1, ind] += 1
                if k < 5:
                    k100[0] += 1
                    k100_rel[0, ind] += 1
                if k < 10:
                    k100[1] += 1
                    k100_rel[1, ind] += 1
                if k < 15:
                    k100[2] += 1
                    k100_rel[2, ind] += 1
                if k < 20:
                    k100[3] += 1
                    k100_rel[3, ind] += 1
            if r in rel_200:
                rel_200_num += 1
                if r not in hash_200:
                    hash_200[r] = len(hash_200)
                ind = hash_200[r]
                k200_rel[-1, ind] += 1
                if k < 5:
                    k200[0] += 1
                    k200_rel[0, ind] += 1
                if k < 10:
                    k200[1] += 1
                    k200_rel[1, ind] += 1
                if k < 15:
                    k200[2] += 1
                    k200_rel[2, ind] += 1
                if k < 20:
                    k200[3] += 1
                    k200_rel[3, ind] += 1
    k100 = k100 / rel_100_num * 100
    k200 = k200 / rel_200_num * 100
    logger.info("Micro: ")
    logger.info("For ins num < 100, hits@5: {0:.1f}, hits@10: {1:.1f}, hits@15: {2:.1f}, hits@20: {3:.1f}".format(k100[0], k100[1], k100[2], k100[3]))
    logger.info("For ins num < 200, hits@5: {0:.1f}, hits@10: {1:.1f}, hits@15: {2:.1f}, hits@20: {3:.1f}".format(k200[0], k200[1], k200[2], k200[3]))
    k100_rel = k100_rel[:, :len(hash_100)]
    k200_rel = k200_rel[:, :len(hash_200)]
    k100_rel /= k100_rel[-1, :]
    k200_rel /= k200_rel[-1, :]
    k100_rel = k100_rel.mean(1) * 100
    k200_rel = k200_rel.mean(1) * 100
    logger.info("Macro: ")
    logger.info("For ins num < 100, hits@5: {0:.1f}, hits@10: {1:.1f}, hits@15: {2:.1f}, hits@20: {3:.1f}".format(k100_rel[0], k100_rel[1], k100_rel[2], k100_rel[3]))
    logger.info("For ins num < 200, hits@5: {0:.1f}, hits@10: {1:.1f}, hits@15: {2:.1f}, hits@20: {3:.1f}".format(k200_rel[0], k200_rel[1], k200_rel[2], k200_rel[3]))


if __name__ == '__main__':
    opt = config()
    setup_seed(opt['seed'])
    logger = get_logger(opt['log_dir'], 'eval_result.log')
    loader = data_loader(opt['test'], logger, opt, shuffle=False, training=False)
    pn_1_loader = data_loader(opt['test'], logger, opt, shuffle=False, training=False, pn_mode='one')
    pn_2_loader = data_loader(opt['test'], logger, opt, shuffle=False, training=False, pn_mode='two')
    pn_all_loader = data_loader(opt['test'], logger, opt, shuffle=False, training=False, pn_mode='all')
    pn_loaders = [pn_1_loader, pn_2_loader, pn_all_loader]
    test(loader, pn_loaders, logger, opt)

