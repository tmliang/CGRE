import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from collections import Counter
from tqdm import tqdm


class Bag:
    def __init__(self):
        self.word = []
        self.pos1 = []
        self.pos2 = []
        self.ent1 = []
        self.ent2 = []
        self.mask = []
        self.length = []
        self.type = []
        self.rel = []


class Dataset(data.Dataset):
    def _preprocess(self, word2id):
        self.logger.debug("Processed data does not exist, preprocessing...")
        # Load files
        ori_data = json.load(open(self.data_dir))
        # Sort data by entities and relations
        ori_data.sort(key=lambda a: a['sub']['id'] + '#' + a['obj']['id'] + '#' + a['rel'])
        # Pre-process data
        last_bag = None
        self.data = []
        bag = Bag()
        for ins in tqdm(ori_data):
            _rel = self.rel2id[ins['rel']] if ins['rel'] in self.rel2id else self.rel2id['NA']
            if self.training:
                cur_bag = (ins['sub']['id'], ins['obj']['id'], str(_rel))  # used for train
            else:
                cur_bag = (ins['sub']['id'], ins['obj']['id'])  # used for test

            if cur_bag != last_bag:
                if last_bag is not None:
                    self.data.append(bag)
                    bag = Bag()
                last_bag = cur_bag
            # rel label
            bag.rel.append(_rel)

            # type
            _type = [self.type2id[ins['sub']['type']], self.type2id[ins['obj']['type']]]
            bag.type.append(_type)

            # word
            words = ins['text'].split()
            _ids = [word2id[word] if word in word2id else word2id['[UNK]'] for word in words]
            _ids = _ids[:self.max_length]
            _ids.extend([word2id['[PAD]'] for _ in range(self.max_length - len(words))])
            bag.word.append(_ids)

            # sentence length
            _length = min(len(words), self.max_length)
            bag.length.append(_length)

            # ent
            ent1 = ins['sub']['name']
            ent2 = ins['obj']['name']
            _ent1 = word2id[ent1] if ent1 in word2id else word2id['[UNK]']
            _ent2 = word2id[ent2] if ent2 in word2id else word2id['[UNK]']
            bag.ent1.append(_ent1)
            bag.ent2.append(_ent2)

            # pos
            p1 = min(words.index(ent1) if ent1 in words else 0, self.max_length - 1)
            p2 = min(words.index(ent2) if ent2 in words else 0, self.max_length - 1)
            _pos1 = np.arange(self.max_length) - p1 + self.max_pos_length
            _pos2 = np.arange(self.max_length) - p2 + self.max_pos_length
            _pos1[_pos1 > 2 * self.max_pos_length] = 2 * self.max_pos_length
            _pos2[_pos2 > 2 * self.max_pos_length] = 2 * self.max_pos_length
            _pos1[_pos1 < 0] = 0
            _pos2[_pos2 < 0] = 0
            _pos1[_length:] = 2 * self.max_pos_length + 1
            _pos2[_length:] = 2 * self.max_pos_length + 1
            bag.pos1.append(_pos1)
            bag.pos2.append(_pos2)

            # mask
            p1, p2 = sorted((p1, p2))
            _mask = np.zeros(self.max_length, dtype=np.long)
            _mask[p2 + 1: _length] = 3
            _mask[p1 + 1: p2 + 1] = 2
            _mask[:p1 + 1] = 1
            bag.mask.append(_mask)

        # append the last bag
        if last_bag is not None:
            self.data.append(bag)

        self.logger.debug("Finish pre-processing")
        self.logger.debug("Storing processed files...")
        pickle.dump(self.data, open(os.path.join(self.processed_data_dir, self.file_name.split(".")[0]+'.pkl'), 'wb'))
        self.logger.debug("Finish storing")

    def __init__(self, file_name, logger, opt, training=True, pn_mode=None):
        super().__init__()
        self.file_name = file_name
        self.max_length = opt['max_length']
        self.max_pos_length = opt['max_pos_length']
        self.processed_data_dir = opt['processed_data_dir']
        self.data_dir = os.path.join(opt['root'], self.file_name)
        self.rel2id = json.load(open(os.path.join(opt['root'], opt['rel2id'])))
        self.word_vec_dir, word2id = self.init_word(os.path.join(opt['root'], opt['vec_dir']))
        self.type2id = json.load(open(os.path.join(opt['root'], opt['type2id'])))
        self.graph = json.load(open(os.path.join(opt['root'], opt['graph'])))
        self.logger = logger
        self.training = training
        self.pn_mode = pn_mode
        if not os.path.exists(opt['save_dir']):
            os.mkdir(opt['save_dir'])
        try:
            self.logger.debug("Trying to load processed data")
            self.data = pickle.load(open(os.path.join(self.processed_data_dir, self.file_name.split(".")[0]+'.pkl'), 'rb'))
            self.logger.debug("Load successfully")
        except FileNotFoundError:
            self._preprocess(word2id)
        if not training:
            if pn_mode is not None:
                logger.info("Prepocessing Test set in mode %s..." % pn_mode.capitalize())
                self.data = [d for d in self.data if len(d.word) > 1]
            else:
                logger.info("Prepocessing Test set...")
                self.rel_100, self.rel_200 = self.long_tail(os.path.join(opt['root'], opt['train']))
        else:
            logger.info("Prepocessing Training set...")
        self.rel_num = len(self.rel2id)
        self.type_num = len(self.type2id)
        self.batch_num = np.ceil(len(self.data) / opt['batch_size'])
        self.edge, self.constraint = self.build_graph()
        self.logger.info("total bag nums: %d\n" % (self.__len__()))

    def init_word(self, vec_dir):
        if not os.path.isdir(self.processed_data_dir):
            os.mkdir(self.processed_data_dir)
        word2id_dir = os.path.join(self.processed_data_dir, 'word2id.json')
        word_vec_dir = os.path.join(self.processed_data_dir, 'word_vec.npy')
        try:
            word2id = json.load(open(word2id_dir))
        except FileNotFoundError:
            f = open(vec_dir)
            num, dim = [int(x) for x in f.readline().split()[:2]]
            word2id = {}
            word_vec = np.zeros([num+2, dim], dtype=np.float32)
            for line in f.readlines():
                line = line.strip().split()
                word_vec[len(word2id)] = np.array(line[1:])
                word2id[line[0].lower()] = len(word2id)
            f.close()
            word_vec[len(word2id)] = np.random.randn(dim) / np.sqrt(dim)
            word2id['[UNK]'] = len(word2id)
            word2id['[PAD]'] = len(word2id)
            np.save(word_vec_dir, word_vec)
            json.dump(word2id, open(word2id_dir, 'w'))
        return word_vec_dir, word2id

    def long_tail(self, train_dir):
        try:
            self.logger.debug("Trying to load long tail data...")
            long_tail = json.load(open(os.path.join(self.processed_data_dir, 'long_tail.json')))
        except FileNotFoundError:
            self.logger.debug("Long tail data does not exist, extracting...")
            train_data = json.load(open(train_dir))
            rel_ins = [self.rel2id[ins['rel']] for ins in train_data if (ins['rel'] != 'NA' and ins['rel'] in self.rel2id)]
            stat = Counter(rel_ins)
            rel_100 = [k-1 for k, v in stat.items() if v < 100]  # k-1 because 'NA' is removed
            rel_200 = [k-1 for k, v in stat.items() if v < 200]
            long_tail = {"rel_100": rel_100, "rel_200": rel_200}
            json.dump(long_tail, open(os.path.join(self.processed_data_dir, 'long_tail.json'), 'w'))
        return long_tail["rel_100"], long_tail["rel_200"]

    def build_graph(self):
        self.logger.debug("Building Graph...")
        sub = []
        obj = []
        constraint = [[[], []] for _ in range(self.rel_num)]
        for rel, types in self.graph.items():
            r = self.rel2id[rel]
            r_node = r + self.type_num  # rel id -> node id
            for type in types:
                h, t = self.type2id[type[0]], self.type2id[type[1]]
                # h -> r -> t
                sub += [h, r_node]
                obj += [r_node, t]
                constraint[r][0].append(h)
                constraint[r][1].append(t)
            constraint[r] = torch.tensor(constraint[r], dtype=torch.long)
        # for NA
        for i in range(self.type_num):
            sub += [i, self.type_num]
            obj += [self.type_num, i]
            constraint[0][0].append(i)
            constraint[0][1].append(i)
        constraint[0] = torch.tensor(constraint[0], dtype=torch.long)
        edge = torch.tensor([sub, obj], dtype=torch.long)
        if torch.cuda.is_available():
            edge = edge.cuda()
            for i in range(self.rel_num):
                constraint[i] = constraint[i].cuda()
        self.logger.debug("Finish building")
        return edge, constraint

    def rel_weight(self):
        self.logger.debug("Calculating the n-class weight")
        rel_ins = []
        for bag in self.data:
            rel_ins.extend(bag.rel)
        stat = Counter(rel_ins)
        class_weight = torch.ones(self.rel_num, dtype=torch.float32)
        for k, v in stat.items():
            class_weight[k] = 1. / v**0.05
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()
        return class_weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        bag = self.data[index]
        sen_num = len(bag.word)
        select = torch.arange(sen_num)
        if not self.training and self.pn_mode is not None:
            if self.pn_mode == 'one' and sen_num > 1:
                select = torch.tensor(np.random.choice(sen_num, 1), dtype=torch.long)
            elif self.pn_mode == 'two' and sen_num > 2:
                select = torch.tensor(np.random.choice(sen_num, 2, replace=False), dtype=torch.long)
        word = torch.tensor(bag.word, dtype=torch.long)[select]
        pos1 = torch.tensor(bag.pos1, dtype=torch.long)[select]
        pos2 = torch.tensor(bag.pos2, dtype=torch.long)[select]
        ent1 = torch.tensor(bag.ent1, dtype=torch.long)[select]
        ent2 = torch.tensor(bag.ent2, dtype=torch.long)[select]
        mask = torch.tensor(bag.mask, dtype=torch.long)[select]
        type = torch.tensor(bag.type, dtype=torch.long)[select]
        if self.training:
            rel = torch.tensor(bag.rel[0], dtype=torch.long)
        else:
            rel = torch.zeros(len(self.rel2id), dtype=torch.long)
            for i in set(bag.rel):
                rel[i] = 1
        return word, pos1, pos2, ent1, ent2, mask, type, rel


def collate_fn(X):
    X = list(zip(*X))
    word, pos1, pos2, ent1, ent2, mask, type, rel = X
    scope = []
    ind = 0
    for w in word:
        scope.append((ind, ind + len(w)))
        ind += len(w)
    scope = torch.tensor(scope, dtype=torch.long)
    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)
    pos2 = torch.cat(pos2, 0)
    ent1 = torch.cat(ent1, 0)
    ent2 = torch.cat(ent2, 0)
    mask = torch.cat(mask, 0)
    type = torch.cat(type, 0)
    rel = torch.stack(rel)
    return word, pos1, pos2, ent1, ent2, mask, type, scope, rel


def data_loader(data_file, logger, opt, shuffle, training=True, pn_mode=None, num_workers=4):
    dataset = Dataset(data_file, logger, opt, training, pn_mode)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt['batch_size'],
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader