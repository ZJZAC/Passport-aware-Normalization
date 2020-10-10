import os

import torch
import torch.optim as optim

import passport_generator
from dataset import prepare_dataset, prepare_wm
from experiments.base import Experiment
from experiments.trainer import Trainer
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from experiments.utils import construct_passport_kwargs
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.layers.conv2d import ConvBlock
from models.resnet_normal import ResNet18
from models.resnet_passport_private import ResNet18Private


import shutil
import matplotlib.pyplot as plt
from experiments.logger import  Logger, savefig
from prun import test,test_signature,pruning_resnet
import numpy as np

#for ambiguity
from amb_attack import train_maximize, test_fake
import pandas as pd
import torch.nn as nn
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
import torch.nn.functional as F
import numpy as np
import shutil


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)

class ClassificationPrivateExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.in_channels = 1 if self.dataset == 'mnist' else 3
        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256
        }[self.dataset]

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

        self.train_data, self.valid_data = prepare_dataset(self.args)
        self.wm_data = None

        if self.use_trigger_as_passport:
            self.passport_data = prepare_wm('data/trigger_set/pics')
        else:
            self.passport_data = self.valid_data

        if self.train_backdoor:
            self.wm_data = prepare_wm('data/trigger_set/pics')

        self.construct_model()

        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        self.trainer = TrainerPrivate(self.model, optimizer, scheduler, self.device)

        if self.is_tl:
            self.finetune_load()
        else:
            self.makedirs_or_load()

    def construct_model(self):
        def setup_keys():
            if self.key_type != 'random':
                if self.arch == 'alexnet':
                    pretrained_model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
                else:
                    pretrained_model = ResNet18(num_classes=self.num_classes, norm_type=self.norm_type)
                pretrained_model.load_state_dict(torch.load(self.pretrained_path))
                pretrained_model = pretrained_model.to(self.device)
                self.setup_keys(pretrained_model)

        passport_kwargs = construct_passport_kwargs(self)
        self.passport_kwargs = passport_kwargs

        if self.arch == 'alexnet':
            model = AlexNetPassportPrivate(self.in_channels, self.num_classes, passport_kwargs)
        else:
            model = ResNet18Private(num_classes=self.num_classes, passport_kwargs=passport_kwargs)

        self.model = model.to(self.device)

        setup_keys()

    def setup_keys(self, pretrained_model):
        if self.key_type != 'random':
            n = 1 if self.key_type == 'image' else 20  # any number

            key_x, x_inds = passport_generator.get_key(self.passport_data, n)
            key_x = key_x.to(self.device)
            key_y, y_inds = passport_generator.get_key(self.passport_data, n)
            key_y = key_y.to(self.device)

            passport_generator.set_key(pretrained_model, self.model,
                                       key_x, key_y)

    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        best_file = os.path.join(self.logdir, 'best.txt')

        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        for ep in range(1, self.epochs + 1):
            train_metrics = self.trainer.train(ep, self.train_data, self.wm_data)
            print(f'Sign Detection Accuracy: {train_metrics["sign_acc"] * 100:6.4f}')

            valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

            wm_metrics = {}
            if self.train_backdoor:
                wm_metrics = self.trainer.test(self.wm_data, 'WM Result')

            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'wm_{key}'] = wm_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_total_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_total_acc']
                best_ep = ep
                self.save_model('best.pth')

            self.save_last_model()

            f = open(best_file,'a')
            f.write(str(best_acc) + "\n")
            print(str(wm_metrics) + '\n',file=f)
            print(str(metrics) + '\n',file=f)
            f.write( "\n")
            f.write("best epoch: %s"%str(best_ep) + '\n')
            f.flush()


    def evaluate(self):
        self.trainer.test(self.valid_data)

    def transfer_learning(self):
        if not self.is_tl:
            raise Exception('Please run with --transfer-learning')

        if self.tl_dataset == 'caltech-101':
            self.num_classes = 101
        elif self.tl_dataset == 'cifar100':
            self.num_classes = 100
        elif self.tl_dataset == 'caltech-256':
            self.num_classes = 257
        else:  # cifar10
            self.num_classes = 10

        # load clone model
        print('Loading clone model')
        if self.arch == 'alexnet':
            tl_model = AlexNetNormal(self.in_channels,
                                     self.num_classes,
                                     self.norm_type)
        else:
            tl_model = ResNet18(num_classes=self.num_classes,
                                norm_type=self.norm_type)

        ##### load / reset weights of passport layers for clone model #####
        try:
            tl_model.load_state_dict(self.model.state_dict())
        except:
            print('Having problem to direct load state dict, loading it manually')
            if self.arch == 'alexnet':
                for tl_m, self_m in zip(tl_model.features, self.model.features):
                    try:
                        tl_m.load_state_dict(self_m.state_dict())
                    except:
                        print(
                            'Having problem to load state dict usually caused by missing keys, load by strict=False')
                        tl_m.load_state_dict(self_m.state_dict(), False)  # load conv weight, bn running mean
                        tl_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                        tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

            else:
                passport_settings = self.passport_config
                for l_key in passport_settings:  # layer
                    if isinstance(passport_settings[l_key], dict):
                        for i in passport_settings[l_key]:  # sequential
                            for m_key in passport_settings[l_key][i]:  # convblock
                                tl_m = tl_model.__getattr__(l_key)[int(i)].__getattr__(m_key)  # type: ConvBlock
                                self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)

                                try:
                                    tl_m.load_state_dict(self_m.state_dict())
                                except:
                                    print(f'{l_key}.{i}.{m_key} cannot load state dict directly')
                                    tl_m.load_state_dict(self_m.state_dict(), False)
                                    tl_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                                    tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

                    else:
                        tl_m = tl_model.__getattr__(l_key)
                        self_m = self.model.__getattr__(l_key)

                        try:
                            tl_m.load_state_dict(self_m.state_dict())
                        except:
                            print(f'{l_key} cannot load state dict directly')
                            tl_m.load_state_dict(self_m.state_dict(), False)
                            tl_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                            tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

        tl_model.to(self.device)
        print('Loaded clone model')

        # tl scheme setup
        if self.tl_scheme == 'rtal':
            # rtal = reset last layer + train all layer
            # ftal = train all layer
            try:
                tl_model.classifier.reset_parameters()
            except:
                tl_model.linear.reset_parameters()

        optimizer = optim.SGD(tl_model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        tl_trainer = Trainer(tl_model,
                             optimizer,
                             scheduler,
                             self.device)
        tester = TesterPrivate(self.model,
                               self.device)

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True
        best_acc = 0
        best_file = os.path.join(self.logdir, 'best.txt')
        best_ep = 1

        for ep in range(1, self.epochs + 1):
            train_metrics = tl_trainer.train(ep, self.train_data)
            valid_metrics = tl_trainer.test(self.valid_data)

            ##### load transfer learning weights from clone model  #####
            try:
                self.model.load_state_dict(tl_model.state_dict())
            except:
                if self.arch == 'alexnet':
                    for tl_m, self_m in zip(tl_model.features, self.model.features):
                        try:
                            self_m.load_state_dict(tl_m.state_dict())
                        except:
                            self_m.load_state_dict(tl_m.state_dict(), False)
                else:
                    passport_settings = self.passport_config
                    for l_key in passport_settings:  # layer
                        if isinstance(passport_settings[l_key], dict):
                            for i in passport_settings[l_key]:  # sequential
                                for m_key in passport_settings[l_key][i]:  # convblock
                                    tl_m = tl_model.__getattr__(l_key)[int(i)].__getattr__(m_key)
                                    self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)

                                    try:
                                        self_m.load_state_dict(tl_m.state_dict())
                                    except:
                                        self_m.load_state_dict(tl_m.state_dict(), False)
                        else:
                            tl_m = tl_model.__getattr__(l_key)
                            self_m = self.model.__getattr__(l_key)

                            try:
                                self_m.load_state_dict(tl_m.state_dict())
                            except:
                                self_m.load_state_dict(tl_m.state_dict(), False)

            wm_metrics = tester.test_signature()
            L = len(wm_metrics)
            S = sum(wm_metrics.values())
            pri_sign = S/L

            if self.train_backdoor:
                backdoor_metrics = tester.test(self.wm_data, 'Old WM Accuracy')

            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'old_wm_{key}'] = wm_metrics[key]
            if self.train_backdoor:
                for key in backdoor_metrics: metrics[f'backdoor_{key}'] = backdoor_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')
                self.save_model(f'tl-epoch-{ep}.pth', tl_model)

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')
                self.save_model('tl-best.pth', tl_model)
                best_ep = ep

            self.save_last_model()
            f = open(best_file,'a')
            print(str(wm_metrics) + '\n', file=f)
            print(str(metrics) + '\n', file=f)
            f.write('Bset ACC %s'%str(best_acc) + "\n")
            print('Private Sign Detction:',str(pri_sign) + '\n', file=f)
            f.write("best epoch: %s"%str(best_ep) + '\n')
            f.flush()

    def fake_attack(self):
        epochs = 100
        lr = 0.01
        device = self.device

        loadpath = self.logdir
        task_name = loadpath.split('/')[-2]
        loadpath_all = loadpath + '/models/best.pth'
        sd = torch.load(loadpath_all)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)
        logdir = '/data-x/g12/zhangjie/DeepIPR/baseline/passport_attack/' + task_name + '/' + self.type
        os.makedirs(logdir, exist_ok=True)
        best_file = os.path.join(logdir, 'best.txt')
        log_file = os.path.join(logdir, 'log.txt')
        lf = open(log_file, 'a')
        shutil.copy('amb_attack.py', str(logdir) + "/amb_attack.py")

        for param in self.model.parameters():
            param.requires_grad_(False)


        passblocks = []
        origpassport = []
        fakepassport = []

        for m in self.model.modules():
            if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):

                passblocks.append(m)

                keyname = 'key_private'
                skeyname = 'skey_private'

                key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()
                origpassport.append(key.to(device))
                origpassport.append(skey.to(device))

                m.__delattr__(keyname) #删除属性
                m.__delattr__(skeyname)

                #fake like random onise
                if  'fake2-' in self.type :
                    shape = []
                    x = torch.cuda.FloatTensor(shape)
                    noise1 = torch.randn(*key.size(), out=x)
                    noise2 = torch.randn(*key.size(), out=x)

                    torch.randn(*key.size())
                    m.register_parameter(keyname, nn.Parameter( noise1 ))
                    m.register_parameter(skeyname, nn.Parameter( noise2 ))


                # fake slightly modify ori
                else:
                    shape = []
                    x = torch.cuda.FloatTensor(shape)
                    noise1 = torch.randn(*key.size(), out=x)
                    noise2 = torch.randn(*key.size(), out=x)

                    torch.randn(*key.size())
                    m.register_parameter(keyname, nn.Parameter(key.clone().cuda() + noise1 * 0.001))
                    m.register_parameter(skeyname, nn.Parameter(skey.clone().cuda() + noise2 * 0.001))

                fakepassport.append(m.__getattr__(keyname))
                fakepassport.append(m.__getattr__(skeyname))


        if self.flipperc != 0:
            print(f'Reverse {self.flipperc * 100:.2f}% of binary signature')
            for m in passblocks:
                mflip = self.flipperc

                oldb = m.sign_loss_private.b
                newb = oldb.clone()

                npidx = np.arange(len(oldb))   #bit 长度
                randsize = int(oldb.view(-1).size(0) * mflip)
                randomidx = np.random.choice(npidx, randsize, replace=False) #随机选择

                newb[randomidx] = oldb[randomidx] * -1  # reverse bit  进行翻转


                m.sign_loss_private.set_b(newb)

        self.model.to(device)

        optimizer = torch.optim.SGD(fakepassport,
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=0.0005)

        scheduler = None
        criterion = nn.CrossEntropyLoss()

        history = []

        def run_cs():  #计算余弦相似性
            cs = []

            for d1, d2 in zip(origpassport, fakepassport):
                d1 = d1.view(d1.size(0), -1)
                d2 = d2.view(d2.size(0), -1)

                cs.append(F.cosine_similarity(d1, d2).item())

            return cs

        print('Before training')
        print('Before training', file = lf)

        res = {}
        valres = test_fake(self.model, criterion, self.valid_data, device)
        for key in valres: res[f'valid_{key}'] = valres[key]

        print(res)
        print(res,file=lf)
        # sys.exit(0)

        with torch.no_grad():
            cs = run_cs()

            mseloss = 0
            for l, r in zip(origpassport, fakepassport):
                mse = F.mse_loss(l, r)
                mseloss += mse.item()
            mseloss /= len(origpassport)

        print(f'MSE of Real and Maximize passport: {mseloss:.4f}')
        print(f'MSE of Real and Maximize passport: {mseloss:.4f}', file=lf)
        print(f'Cosine Similarity of Real and Maximize passport: {sum(cs) / len(origpassport):.4f}')
        print(f'Cosine Similarity of Real and Maximize passport: {sum(cs) / len(origpassport):.4f}', file=lf)
        print()

        res['epoch'] = 0
        res['cosine_similarity'] = cs
        res['flipperc'] = self.flipperc
        res['train_mseloss'] = mseloss

        history.append(res)

        torch.save({'origpassport': origpassport,
                    'fakepassport': fakepassport,
                    'state_dict': self.model.state_dict()},
                    f'{logdir}/{self.arch}-v3-last-{self.dataset}-{self.type}-{self.flipperc:.1f}-e0.pth')

        best_acc = 0
        best_sign_acc = 0
        best_ep = 0

        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()

            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            print(f'Epoch {ep:3d}:')
            print(f'Epoch {ep:3d}:',file=lf)
            print('Training')
            trainres = train_maximize(origpassport, fakepassport, self.model, optimizer, criterion, self.train_data, device, self.type)

            print('Testing')
            print('Testing',file=lf)
            valres = test_fake(self.model, criterion, self.valid_data, device)

            print(valres,file=lf)
            print('\n',file=lf)

            # if best_acc < valres['acc']:
            if best_sign_acc < valres['signacc']:
                print(f'Found best sign acc at epoch {ep}\n')
                # best_acc = valres['acc']
                best_sign_acc = valres['signacc']
                best_ep = ep

            f = open(best_file,'a')
            f.write(str(best_sign_acc) + '\n')
            f.write("best sing aca epoch: %s"%str(best_ep) + '\n')
            f.flush()

            res = {}

            for key in trainres: res[f'train_{key}'] = trainres[key]
            for key in valres: res[f'valid_{key}'] = valres[key]
            res['epoch'] = ep
            res['flipperc'] = self.flipperc

            with torch.no_grad():
                cs = run_cs()
                res['cosine_similarity'] = cs

            print(f'Cosine Similarity of Real and Maximize passport: '
                  f'{sum(cs) / len(origpassport):.4f}')
            print()

            print(f'Cosine Similarity of Real and Maximize passport: '
                  f'{sum(cs) / len(origpassport):.4f}'+'\n', file=lf)
            lf.flush()

            history.append(res)

            torch.save({'origpassport': origpassport,
                        'fakepassport': fakepassport,
                        'state_dict': self.model.state_dict()},
                        f'{logdir}/{self.arch}-v3-last-{self.dataset}-{self.type}-{self.flipperc:.1f}-e{ep}.pth')


            histdf = pd.DataFrame(history)
        histdf.to_csv(f'{logdir}/{self.arch}-v3-history-{self.dataset}-{self.type}-{self.flipperc:.1f}.csv')

    def pruning(self):
        device = self.device
        logdir = f'/data-x/g12/zhangjie/DeepIPR/baseline/{self.arch}_{self.dataset}_v3_{self.tag}/1'

        prun_dir = logdir + '/prun'
        if not os.path.exists(prun_dir):
            os.mkdir(prun_dir)
        shutil.copy('prun.py', str(prun_dir) + "/prun.py")

        title = ''  #
        txt_pth = os.path.join(prun_dir, 'log_prun.txt')
        logger_prun = Logger(txt_pth, title=title)
        logger_prun.set_names(['Model for Releasing  ', 'Model for Verification ', 'Trigger','Signature'])

        self.train_data, self.valid_data = prepare_dataset(self.args)

        sd = torch.load(logdir + '/models/best.pth')
        print(logdir + '/models/best.pth')

        self.model.load_state_dict(sd)
        self.model.to(self.device)
        test(self.model, device, self.valid_data, msg='Ori_pub Result', ind=0)
        test(self.model, device, self.valid_data, msg='Pri_pub Result', ind=1)

        for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            sd = torch.load(logdir + '/models/best.pth')
            self.model.load_state_dict(sd)
            self.model.to(self.device)
            pruning_resnet(self.model, perc)
            res = {}
            res_wm = {}

            self.wm_data = prepare_wm('data/trigger_set/pics')

            res['perc'] = perc
            res['pub_ori'] = test(self.model, device, self.valid_data, msg='pruning %s percent Public Result' % perc, ind=0)
            res['pri_ori'] = test(self.model, device, self.valid_data, msg='pruning %s percent Private Result' % perc, ind=1)
            _, res['pri_sign_acc'] = test_signature(self.model)

            res_wm['pri_ori'] = test(self.model, device, self.wm_data, msg='pruning %s percent Pri_Trigger Result' % perc, ind=1)

            pub_acc = res['pub_ori']['acc']
            pri_acc = res['pri_ori']['acc']
            pri_acc_wm = res_wm['pri_ori']['acc']
            pri_sign_acc = res['pri_sign_acc'] * 100
            logger_prun.append([pub_acc, pri_acc, pri_acc_wm, pri_sign_acc])

        file = open(txt_pth, "r")

        name = file.readline()
        names = name.rstrip().split('\t')  # 标题分开
        numbers = {}
        for _, name in enumerate(names):
            numbers[name] = []

        for line in file:
            numbers_clm = line.rstrip().split('\t')

            for clm in range(0, len(numbers_clm)):
                numbers[names[clm]].append(numbers_clm[clm])

        #########plot####################

        plt.figure()
        names_legend = []
        for i, name in enumerate(names):

            x = np.arange(len(numbers[name]))
            num_float = []
            for num in numbers[name]:
                num_float.append(float(num))

            names_legend.append(name)
            x = np.arange(len(numbers[name])) * 10  # start with 10%
            plt.plot(x, num_float)

        plt.legend([name for name in names_legend], fontsize=15)

        plt.grid(True)
        plt.ylabel('Accuracy (%)', fontsize=20)
        plt.xlabel('Pruning rate (%)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        save_name = prun_dir + '/prun.eps'
        # plt.savefig(save_name)
        savefig(save_name)