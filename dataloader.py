import os
import random
import collections

import dgl
import torch
import numpy as np
import pandas as pd


class DataLoaderKGAT(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size

        data_dir = os.path.join(args.data_dir, args.data_name)
        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')
        kg_file = os.path.join(data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(test_file)
        self.statistic_cf()

        kg_data = self.load_kg(kg_file)
        self.construct_data(kg_data)

        self.print_info(logging)
        self.train_graph = self.create_graph(self.kg_train_data, self.n_users_entities)
        self.test_graph = self.create_graph(self.kg_test_data, self.n_users_entities)

        if self.use_pretrain == 1:
            self.load_pretrained_data()


    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user.append(user_id)   #user u跟几个item有交互就append几次u_id
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])  #交互数目
        self.n_cf_test = len(self.cf_test_data[0])


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()   #去重
        return kg_data


    def construct_data(self, kg_data):   #将h和t互换(h变t，t变h)后加入到知识图谱中，然后在kg加入user item交互的关系(也是要将互换)，互换是因为关系是有方向的。
        # plus inverse kg data
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id
        kg_data['r'] += 2   #加2是因为把user和item的交互关系（采用和被采用算是两种关系）加入，形成ckg
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities
        # print('****************cf_train_data[0]**************')
        # print(self.cf_train_data[0])

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32)) 
        #user_id+实体数目
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))
        # print('****************cf_train_data[0]**************')
        # print(self.cf_train_data[0])
        
        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}  
        #k是user_id v是与user有交互的items列表
        # np.unique()该函数是去除数组中的重复数字，并进行排序之后输出。
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        interact_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        interact_train_data['h'] = self.cf_train_data[0]
        interact_train_data['t'] = self.cf_train_data[1]

        reverse_interact_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_interact_train_data['h'] = self.cf_train_data[1]
        reverse_interact_train_data['t'] = self.cf_train_data[0]

        interact_test_data = pd.DataFrame(np.zeros((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        interact_test_data['h'] = self.cf_test_data[0]
        interact_test_data['t'] = self.cf_test_data[1]

        reverse_interact_test_data = pd.DataFrame(np.ones((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_interact_test_data['h'] = self.cf_test_data[1]
        reverse_interact_test_data['t'] = self.cf_test_data[0]

        self.kg_train_data = pd.concat([kg_data, interact_train_data, reverse_interact_train_data], ignore_index=True)
        self.kg_test_data = pd.concat([kg_data, interact_test_data, reverse_interact_test_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_test = len(self.kg_test_data)

        # construct kg dict
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        #iterrows() 是在数据框中(DataFrame)的行进行迭代的一个生成器，它返回每行的索引及一个包含行本身的对象。
        # print('***************self.kg_train_data*************************')
        # print(self.kg_train_data)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            # print('***************row[1]*************************')
            # print(row[1])
            # break
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.test_kg_dict = collections.defaultdict(list)
        self.test_relation_dict = collections.defaultdict(list)
        for row in self.kg_test_data.iterrows():
            h, r, t = row[1]
            self.test_kg_dict[h].append((t, r))
            self.test_relation_dict[r].append((h, t))


    def print_info(self, logging):
        logging.info('n_users:            %d' % self.n_users)
        logging.info('n_items:            %d' % self.n_items)
        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_users_entities:   %d' % self.n_users_entities)
        logging.info('n_relations:        %d' % self.n_relations)

        logging.info('n_cf_train:         %d' % self.n_cf_train)
        logging.info('n_cf_test:          %d' % self.n_cf_test)

        logging.info('n_kg_train:         %d' % self.n_kg_train)
        logging.info('n_kg_test:          %d' % self.n_kg_test)


    def create_graph(self, kg_data, n_nodes):
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.add_edges(kg_data['t'], kg_data['h'])
        g.readonly()
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)  #给图的顶点设置id属性
        g.edata['type'] = torch.LongTensor(kg_data['r'])  #给图的边设置关系(type)属性
        return g