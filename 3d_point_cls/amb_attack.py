"""
Author: Benny
Date: Nov 2019
"""
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from pprint import pprint
from data import getData
import time
from utils import  Logger,  savefig
from models.loss.sign_loss import SignLoss
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--seed', type=int, default=0, help=' seed value [default: 0]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')

parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training [default: 200]')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
parser.add_argument('--beta', default=1, type=float, help='weights of ori loss [default: 1]')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--num_class', type=int, default=40, help='Class Number [default: 40]')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
parser.add_argument('--remark', type=str, default=None, help='exp remark')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')


parser.add_argument('--type', default='fake2-10', help='fake key type, fake2, fake3_100S')
parser.add_argument('--flipperc', default=1, type=float, help='flip percentange 0~1')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01) for reverse engineering')
parser.add_argument('--dataset', type=str, default="shapenet", help='Point Number [default: shapenet, modelnet]')
parser.add_argument('--model', default='pointnet_cls_our_bn', help='model name ')
parser.add_argument('--task', type=str, default='ours', help='exp task')
parser.add_argument('--norm', type=str, default='bn', help='type of normlization [default: P-BN-AK,BN, BNCE, GN, GNCE]')

def test(model, loader, num_class=40, ind=1):
    criterion = nn.NLLLoss().cuda()  #不要对角线loss
    mean_loss = []
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    signloss_meter = 0
    signacc_meter = 0

    model.eval()
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = model(points,ind=ind)
        loss = criterion(pred, target.long())
        mean_loss.append(loss.item() / float(points.size()[0]))
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))

        signloss = torch.tensor(0.).cuda()
        signacc = torch.tensor(0.).cuda()
        count = 0

        for m in model.modules():
            if isinstance(m, SignLoss):
                signloss += m.get_loss()
                signacc += m.get_acc()
                count += 1
        signloss_meter += signloss.item()
        try:
            signacc_meter += signacc.item() / count
        except:
            pass

    signloss = signloss_meter / len(loader)
    signacc = signacc_meter / len(loader)

    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    val_loss = np.mean(mean_loss)


    return val_loss, instance_acc, class_acc, signloss, signacc



def main():
    args = parser.parse_args()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)

    if args.remark != None:
        args.remark = args.remark
    else:
        args.remark = args.dataset + "-" + args.task + "-" + args.norm

    if args.dataset =="shapenet":
        args.num_class=16
    else:
        args.num_class=40

    if 'fake2-' in args.type  :
        args.flipperc = 0
        print('No Flip')
    elif  'fake3-' in args.type:
        args.flipperc = 0
        print('No Flip')

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    if   args.task == 'baseline':
        experiment_dir_root = Path('/data-x/g12/zhangjie/3dIP/baseline')
    else:
        experiment_dir_root = Path('/data-x/g12/zhangjie/3dIP/ours')

    experiment_dir_root.mkdir(exist_ok=True)
    experiment_dir = experiment_dir_root.joinpath('ambiguity_attack')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.remark)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.type)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model") #log name
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)  #log file name
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    if args.dataset == "shapenet":
        trainDataLoader = getData.get_dataLoader(train=True, Shapenet=True, batchsize=args.batch_size)
        testDataLoader = getData.get_dataLoader(train=False,Shapenet=True, batchsize=args.batch_size)
    else:
        trainDataLoader = getData.get_dataLoader(train=True, Shapenet=False, batchsize=args.batch_size)
        testDataLoader = getData.get_dataLoader(train=False,Shapenet=False, batchsize=args.batch_size)

    log_string('Finished ...')
    log_string('Load model ...')

    '''MODEL LOADING'''
    num_class = args.num_class
    MODEL = importlib.import_module(args.model)

    #  copy model file to exp dir
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    shutil.copy('attack2.py', str(experiment_dir))


    classifier = MODEL.get_model(num_class, channel=3).cuda()
    # criterion = MODEL.get_loss().cuda()
    criterion =nn.NLLLoss().cuda()

    sd = experiment_dir_root.joinpath('classification')
    sd.mkdir(exist_ok=True)
    sd = sd.joinpath(str(args.remark))
    sd.mkdir(exist_ok=True)
    sd = sd.joinpath('checkpoints/best_model.pth')


    checkpoint = torch.load(sd)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    for param in classifier.parameters():
        param.requires_grad_(False)

    origpassport = []
    fakepassport = []

    for n, m in classifier.named_modules():
        if n in ['convp1', 'convp2', 'convp3', 'p1', 'p2', 'fc3']:
            key, skey = m.__getattr__('key_private').data.clone(), m.__getattr__('skey_private').data.clone()
            origpassport.append(key.cuda())
            origpassport.append(skey.cuda())

            m.__delattr__('key_private')  # 删除属性
            m.__delattr__('skey_private')

            # fake like random onise
            if 'fake2-' in args.type:
                # fake random
                m.register_parameter('key_private', nn.Parameter(torch.randn(*key.size()) * 0.001, requires_grad=True))
                m.register_parameter('skey_private', nn.Parameter(torch.randn(*skey.size()) * 0.001, requires_grad=True))

            # fake slightly modify ori
            else:
                # fake slightly modify ori
                m.register_parameter('key_private', nn.Parameter(key.clone() + torch.randn(*key.size()) * 0.001, requires_grad=True))
                m.register_parameter('skey_private', nn.Parameter(skey.clone() + torch.randn(*skey.size()) * 0.001, requires_grad=True))

            fakepassport.append(m.__getattr__('key_private'))
            fakepassport.append(m.__getattr__('skey_private'))

            if args.task == 'ours':
                if args.type != 'fake2':

                    for layer in m.fc.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_normal_(layer.weight)

                    for i in m.fc.parameters():
                        i.requires_grad = True

    if args.flipperc != 0:
        log_string(f'Reverse {args.flipperc * 100:.2f}% of binary signature')

        for name, m in classifier.named_modules():
            if name in ['convp1', 'convp2', 'convp3', 'p1', 'p2', 'p3']:
                mflip = args.flipperc
                oldb = m.sign_loss_private.b
                newb = oldb.clone()
                npidx = np.arange(len(oldb))  # bit 长度
                randsize = int(oldb.view(-1).size(0) * mflip)
                randomidx = np.random.choice(npidx, randsize, replace=False)  # 随机选择
                newb[randomidx] = oldb[randomidx] * -1  # reverse bit  进行翻转
                m.sign_loss_private.set_b(newb)


    classifier.cuda()
    optimizer = torch.optim.SGD(fakepassport,
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=0.0005)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    scheduler = None

    def run_cs():
        cs = []

        for d1, d2 in zip(origpassport, fakepassport):
            d1 = d1.view(d1.size(0), -1)
            d2 = d2.view(d2.size(0), -1)

            cs.append(F.cosine_similarity(d1, d2).item())

        return cs

    classifier.train()
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct2 = []
    mean_loss2 = []
    start_epoch = 0

    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        time_start = datetime.datetime.now()
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        optimizer.zero_grad()
        signacc_meter = 0
        signloss_meter = 0

        if scheduler is  not None:
            scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()


            # loss define
            pred2, _ = classifier(points,ind=1)
            loss2 = criterion(pred2, target.long())
            mean_loss2.append(loss2.item() / float(points.size()[0]))
            pred_choice2 = pred2.data.max(1)[1]
            correct2 = pred_choice2.eq(target.long().data).cpu().sum()
            mean_correct2.append(correct2.item() / float(points.size()[0]))

            signacc = torch.tensor(0.).cuda()
            count = 0
            for m in classifier.modules():
                if isinstance(m, SignLoss):
                    signacc += m.get_acc()
                    count += 1
            try:
                signacc_meter += signacc.item() / count
            except:
                pass

            sign_loss = torch.tensor(0.).cuda()
            for m in classifier.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss
            signloss_meter += sign_loss

            loss = loss2
            maximizeloss = torch.tensor(0.).cuda()
            mseloss = torch.tensor(0.).cuda()
            csloss = torch.tensor(0.).cuda()

            for l, r in zip(origpassport, fakepassport):
                mse = mse_criterion(l, r)
                cs = cs_criterion(l.view(1, -1), r.view(1, -1)).mean()
                csloss += cs
                mseloss += mse
                maximizeloss += 1 / mse

            if 'fake2-' in args.type:
                (loss).backward()  # only cross-entropy loss  backward  fake2
            elif 'fake3-' in args.type:
                (loss + maximizeloss).backward()  # csloss do not backward   kafe3

            else:
                (loss + maximizeloss + 1000 * sign_loss).backward()  # csloss  backward   #fake3_S
                # (loss  + 1000 * sign_loss).backward()  # csloss  backward   #fake3_S

            torch.nn.utils.clip_grad_norm_(fakepassport, 2)

            optimizer.step()
            global_step += 1

        signacc = signacc_meter / len(trainDataLoader)
        log_string('Train Sign Accuracy: %f' % signacc)

        signloss = signloss_meter / len(trainDataLoader)
        log_string('Train Sign Loss: %f' % signloss)

        train_instance_acc2 = np.mean(mean_correct2)
        log_string('Train Instance Private Accuracy: %f' % train_instance_acc2)


        with torch.no_grad():
            cs = run_cs()
            log_string(f'Cosine Similarity of Real and Maximize passport: {sum(cs) / len(origpassport):.4f}')
            val_loss2, test_instance_acc2, class_acc2,  singloss2, signacc2  = test(classifier, testDataLoader, num_class= args.num_class, ind =1)

            log_string('Test Instance Private Accuracy: %f, Class Private Accuracy: %f'% (test_instance_acc2, class_acc2))
            log_string('Test Private Sign Accuracy: %f'% (signacc2))


            test_instance_acc =  test_instance_acc2
            class_acc = class_acc2

            if (test_instance_acc >= best_instance_acc):
                best_instance_acc = test_instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Average Accuracy: %f, Class Average Accuracy: %f'% (test_instance_acc, class_acc))
            log_string('Best Instance Average Accuracy: %f, Class Average Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (test_instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_attack_model.pth'
                log_string('Saving at %s'% savepath)
                log_string('best_epoch %s'% str(best_epoch))
                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'origpassport': origpassport,
                    'fakepassport': fakepassport
                }
                torch.save(state, savepath)
            global_epoch += 1


        time_end = datetime.datetime.now()
        time_span_str = str((time_end - time_start).seconds)
        log_string('Epoch time : %s S' % (time_span_str))

    log_string('best_epoch %s' % str(best_epoch))

    logger.info('End of training...')



if __name__ == '__main__':
    main()
