"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
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
from data import getData, getData2
import time
from utils import  Logger,  savefig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


'''PARAMETERS'''
parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--seed', type=int, default=0, help=' seed value [default: 0]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
parser.add_argument('--dataset', type=str, default="modelnet", help='Point Number [default: shapenet, modelnet]')
parser.add_argument('--model', default='pointnet_cls_ori_bn', help='model name [default: pointnet_cls]')
parser.add_argument('--epoch',  default=300, type=int, help='number of epoch in training [default: 200]')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--num_class', type=int, default=16, help='Class Number [default: 40]')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
parser.add_argument('--remark', type=str, default=None, help='exp remark')
parser.add_argument('--task', type=str, default='ori_trigger', help='exp task')
parser.add_argument('--norm', type=str, default='BN', help='type of normlization [default: BN, BNCE, GN, GNCE]')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')

def test(model, loader, num_class=40):
    args = parser.parse_args()
    MODEL = importlib.import_module(args.model)
    criterion = MODEL.get_loss().cuda()
    mean_loss = []
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    model.eval()
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans_feat = model(points)
        loss = criterion(pred, target.long(), trans_feat)
        mean_loss.append(loss.item() / float(points.size()[0]))
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    val_loss = np.mean(mean_loss)
    return val_loss, instance_acc, class_acc


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.remark != None:
        args.remark = args.remark
    else:
        args.remark = args.dataset + "-" + args.task + "-" + args.norm

    if args.dataset =="shapenet":
        args.num_class=16
    else:
        args.num_class=40

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # experiment_dir = Path('./exp/v1/')
    experiment_dir = Path('/data-x/g12/zhangjie/3dIP/exp/v1')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.remark +"_"+ timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG_curve'''
    title = args.dataset + "-" + args.task + "-" + args.norm
    logger_loss = Logger(os.path.join(log_dir, 'log_loss_v1.txt'), title=title)
    logger_loss.set_names([ 'Train Loss', 'Valid Clean Loss', 'Valid Trigger Loss' ])
    logger_acc = Logger(os.path.join(log_dir, 'log_acc_v1.txt'), title=title)
    logger_acc.set_names([  'Train  Acc.',  'Valid Clean Acc.', 'Valid Trigger Acc.'])

    '''LOG'''  #创建log文件
    logger = logging.getLogger("Model") #log的名字
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO) #log的最低等级
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)  #log文件名
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    if args.dataset == "shapenet":
        trainDataLoader = getData.get_dataLoader(train=True, Shapenet=True, batchsize=args.batch_size)
        testDataLoader = getData.get_dataLoader(train=False,Shapenet=True, batchsize=args.batch_size)
        triggerDataLoader = getData2.get_dataLoader(Shapenet=True, batchsize=args.batch_size)
    else:
        trainDataLoader = getData.get_dataLoader(train=True, Shapenet=False, batchsize=args.batch_size)
        testDataLoader = getData.get_dataLoader(train=False,Shapenet=False, batchsize=args.batch_size)
        triggerDataLoader = getData2.get_dataLoader(Shapenet=False, batchsize=args.batch_size)

    wminputs, wmtargets = [], []
    for wm_idx, (wminput, wmtarget) in enumerate(triggerDataLoader):
        wminputs.append(wminput)
        wmtargets.append(wmtarget)

    '''MODEL LOADING'''
    num_class = args.num_class
    MODEL = importlib.import_module(args.model)

    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    shutil.copy('train_1_cls.py', str(experiment_dir))
    shutil.copy('./data/getData.py', str(experiment_dir))
    shutil.copy('./data/getData2.py', str(experiment_dir))
    shutil.copytree('./models/layers', str(experiment_dir)+"/layers")

    classifier = MODEL.get_model(num_class, channel=3).cuda()
    # classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    pprint(classifier)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []
    mean_loss = []


    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        time_start = datetime.datetime.now()
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            wm_id = np.random.randint(len(wminputs))
            points = torch.cat([points, wminputs[(wm_id + batch_id) % len(wminputs)]], dim=0) #随机选择wininputs和inputscat
            target = torch.cat([target, wmtargets[(wm_id + batch_id) % len(wminputs)]], dim=0)

            points = points.data.numpy()
            points = provider.random_point_dropout(points)  #provider是自己写的一个对点云操作的函数，随机dropout，置为第一个点的值
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3]) #点的放缩
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])   #点的偏移
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            mean_loss.append(loss.item() / float(points.size()[0]))

        train_loss = np.mean(mean_loss)
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            val_loss, instance_acc, class_acc = test(classifier, testDataLoader, num_class= args.num_class)
            val_loss2, instance_acc2, class_acc2 = test(classifier, triggerDataLoader, num_class= args.num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Clean Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Clean Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            log_string('Test Trigger Accuracy: %f, Trigger Class Accuracy: %f'% (instance_acc2, class_acc2))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                log_string('best_epoch %s'% str(best_epoch))
                state = {
                    'epoch': best_epoch,
                    'clean instance_acc': instance_acc,
                    'clean class_acc': class_acc,
                    'trigger instance_acc': instance_acc2,
                    'trigger class_acc': class_acc2,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1


        logger_loss.append([train_loss, val_loss, val_loss2])
        logger_acc.append([train_instance_acc, instance_acc, instance_acc2 ])

        time_end = datetime.datetime.now()
        time_span_str = str((time_end - time_start).seconds)
        log_string('Epoch time : %s S' % (time_span_str))

    logger_loss.close()
    logger_loss.plot()
    savefig(os.path.join(log_dir, 'log_loss_v3.eps'))
    logger_acc.close()
    logger_acc.plot()
    savefig(os.path.join(log_dir, 'log_acc_v3.eps'))

    log_string('best_epoch %s' % str(best_epoch))
    logger.info('End of training...')

if __name__ == '__main__':
    main()
