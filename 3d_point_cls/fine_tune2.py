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
from data import getData_ft
import time
from models.pointnet_util import re_initializer_layer, re_initializer_passport_layer
from utils import  Logger,  savefig
from models.loss.sign_loss import SignLoss


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



'''PARAMETERS'''
parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--seed', type=int, default=0, help=' seed value [default: 0]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 300]')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
parser.add_argument('--beta', default=1, type=float, help='weights of ori loss [default: 1]')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--ft_num_class', type=int, default=16, help='Class Number [default: 16,40]')
parser.add_argument('--num_class', type=int, default=40, help='Class Number [default: 40]')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
parser.add_argument('--remark', type=str, default=None, help='exp remark')
parser.add_argument('--ft_remark', type=str, default=None, help='fine-tune exp remark[40to16, 16to40]')

parser.add_argument('--ori_dataset', type=str, default="modelnet", help='Point Number [default: shapenet, modelnet]')
parser.add_argument('--dataset', type=str, default="shapenet", help='Point Number [default: shapenet, modelnet]')
parser.add_argument('--model', default='pointnet_cls_our_bn', help='model name [default:pointnet_cls_all_bn ]')
parser.add_argument('--task', type=str, default='ours', help='exp task')
parser.add_argument('--norm', type=str, default='bn', help='type of normlization [default: BN, BNSE, GN, GNSE]')
parser.add_argument('--rtll', action='store_true', default=False, help='re-train last layer [default: False(rtal)]')
parser.add_argument('--ft_type', type=str, default='RTAL',  help='fine tune strategy [default: RTLL, RTAL]')



def test(model, loader, num_class=40, ind=1):
    args = parser.parse_args()
    MODEL = importlib.import_module(args.model)
    criterion = MODEL.get_loss().cuda()
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
        pred, trans_feat = model(points,ind=ind)
        loss = criterion(pred, target.long(), trans_feat)
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

        if ind ==1 :
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.remark != None:
        args.remark = args.remark
    else:
        args.remark = args.ori_dataset + "-" + args.task + "-" + args.norm

    if args.dataset =="shapenet":
        args.num_class=40
        args.ft_num_class=16
        args.ori_dataset = "modelnet"
    else:
        args.num_class=16
        args.ft_num_class=40
        args.ori_dataset = "shapenet"

    if args.rtll:
        args.ft_type = "RTLL"
    else:
        args.ft_type = "RTAL"

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    if  args.task == 'baseline':
        experiment_dir_root = Path('/data-x/g12/zhangjie/3dIP/baseline')
    else:
        experiment_dir_root = Path('/data-x/g12/zhangjie/3dIP/ours')

    experiment_dir_root.mkdir(exist_ok=True)
    experiment_dir = experiment_dir_root.joinpath('fine-tune')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.remark)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath("ft_"+ args.dataset  +"-"+ args.ft_type + "_"+ timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG_curve'''
    title = ''
    logger_loss = Logger(os.path.join(log_dir, 'log_loss.txt'), title=title)
    logger_loss.set_names([ 'Train AVE Loss', 'Train Public Loss', 'Train Private Loss', 'Valid AVE Loss', 'Valid Public Loss', 'Valid Private Loss'])
    logger_acc = Logger(os.path.join(log_dir, 'log_acc.txt'), title=title)
    logger_acc.set_names([ 'Train AVE Acc.','Train Public Acc.', 'Train Private Acc.', 'Valid AVE Acc.', 'Valid Public Acc.', 'Valid Private Acc.', 'Valid Private Sign Acc.'])

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
    log_string('Load original dataset ...')
    if args.dataset == "shapenet":
        testDataLoader = getData_ft.get_dataLoader(train=False, Shapenet=False, batchsize=args.batch_size)
    else:
        testDataLoader = getData_ft.get_dataLoader(train=False, Shapenet=True, batchsize=args.batch_size)

    log_string('Load finished ...')

    log_string('Load fine tune dataset ...')
    if args.dataset == "shapenet":
        ft_trainDataLoader = getData_ft.get_dataLoader(train=True, Shapenet=True, batchsize=args.batch_size)
        ft_testDataLoader = getData_ft.get_dataLoader(train=False, Shapenet=True, batchsize=args.batch_size)
    else:
        ft_trainDataLoader = getData_ft.get_dataLoader(train=True, Shapenet=False, batchsize=args.batch_size)
        ft_testDataLoader = getData_ft.get_dataLoader(train=False, Shapenet=False, batchsize=args.batch_size)
    log_string('Load finished ...')

    '''MODEL LOADING'''
    num_class = args.num_class
    MODEL = importlib.import_module(args.model)

    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    shutil.copy('fine_tune2.py', str(experiment_dir))


    classifier = MODEL.get_model(num_class, channel=3).cuda()

    # pprint(classifier)

    sd = experiment_dir_root.joinpath('classification')
    sd.mkdir(exist_ok=True)
    sd = sd.joinpath(args.ori_dataset + "-" + args.task + "-" + args.norm )
    sd.mkdir(exist_ok=True)
    sd = sd.joinpath('checkpoints/best_model.pth')


    log_string('pre-trained model chk pth: %s'%sd)
    checkpoint = torch.load(sd)
    model_dict = checkpoint['model_state_dict']
    p_num = get_parameter_number(classifier)
    log_string('Original trainable parameter: %s'%p_num)
    # print(p_num)
    print("best epoch", checkpoint['epoch'])
    classifier.load_state_dict(model_dict)
    classifier.cuda()

    '''TESTING ORIGINAL'''
    logger.info('Test original model...')

    with torch.no_grad():
        _, instance_acc, class_acc, _, _ = test(classifier, testDataLoader, num_class=args.num_class, ind=0)
        _, instance_acc2, class_acc2, signloss, signacc = test(classifier, testDataLoader, num_class=args.num_class, ind=1)
        log_string('Original Instance Public Accuracy: %f, Class Public Accuracy: %f' % (instance_acc, class_acc))
        log_string('Original Instance Private Accuracy: %f, Class Private Accuracy: %f' % (instance_acc2, class_acc2))
        log_string('Private  Sign Accuracy: %f' % (signacc))

    # fine tune the last year
    # classifier, _ = re_initializer_layer(classifier, args.ft_num_class)
    classifier, _ = re_initializer_passport_layer(classifier, args.ft_num_class)
    if args.rtll:
        classifier.freeze_hidden_layers()
    elif args.task == 'ours':
        classifier.freeze_passport_layers()
    else:
        pass

    criterion = MODEL.get_loss().cuda()
    #fine tune param number
    p_num = get_parameter_number(classifier)
    log_string('Fine tune trainable parameter: %s'%p_num)


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
    mean_correct2 = []
    mean_loss = []
    mean_loss1 = []
    mean_loss2 = []

    '''FINR TUNEING'''
    logger.info('Start training of tine tune...')
    start_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        time_start = datetime.datetime.now()
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(ft_trainDataLoader, 0), total=len(ft_trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)  # provider是自己写的一个对点云操作的函数，随机dropout，置为第一个点的值
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  # 点的放缩
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])  # 点的偏移
            points = torch.Tensor(points)
            # target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()


            for ind in range(2):
                if ind == 0:
                    pred, trans_feat = classifier(points,ind=ind)
                    loss1 = criterion(pred, target.long(), trans_feat)
                    mean_loss1.append(loss1.item() / float(points.size()[0]))
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.long().data).cpu().sum()
                    mean_correct.append(correct.item() / float(points.size()[0]))

                else:
                    pred2, trans_feat2 = classifier(points,ind=ind)
                    loss2 = criterion(pred2, target.long(), trans_feat2)
                    mean_loss2.append(loss2.item() / float(points.size()[0]))
                    pred_choice2 = pred2.data.max(1)[1]
                    correct2 = pred_choice2.eq(target.long().data).cpu().sum()
                    mean_correct2.append(correct2.item() / float(points.size()[0]))

            # loss = args.beta * loss1 +loss2
            loss = loss1
            mean_loss.append(loss.item() / float(points.size()[0]))
            # loss = loss2
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        train_instance_acc2 = np.mean(mean_correct2)
        train_instance_acc_ave = (train_instance_acc + train_instance_acc2) / 2
        train_loss = np.mean(mean_loss) / 2
        train_loss1 = np.mean(mean_loss1)
        train_loss2 = np.mean(mean_loss2)
        log_string('FT-Train Instance Public Accuracy: %f' % train_instance_acc)
        log_string('FT-Train Instance Private Accuracy: %f' % train_instance_acc2)

        with torch.no_grad():
            val_loss1, test_instance_acc1, class_acc1, _, _ = test(classifier, ft_testDataLoader, num_class=args.ft_num_class, ind=0)
            val_loss2, test_instance_acc2, class_acc2, signloss, signacc = test(classifier, ft_testDataLoader, num_class=args.ft_num_class,ind=1)
            log_string('FT-Test Instance Public Accuracy: %f, Class Public Accuracy: %f' % (test_instance_acc1, class_acc1))
            log_string('FT-Test Instance Private Accuracy: %f, Class Private Accuracy: %f' % (test_instance_acc2, class_acc2))
            log_string('FT-Test Private  Sign Accuracy: %f' % (signacc))

            # for ind in range(2):
            #     if ind == 0:
            #         val_loss1, test_instance_acc1, class_acc1 = test(classifier, ft_testDataLoader, num_class=args.ft_num_class, ind=0)
            #     else:
            #         val_loss2, test_instance_acc2, class_acc2 = test(classifier, ft_testDataLoader, num_class=args.ft_num_class, ind =1)
            #
            # log_string('FT-Test Instance Public Accuracy: %f, Class Public Accuracy: %f'% (test_instance_acc1, class_acc1))
            # log_string('FT-Test Instance Private Accuracy: %f, Class Private Accuracy: %f'% (test_instance_acc2, class_acc2))

            val_loss = (val_loss1 + val_loss2)/2
            test_instance_acc = (test_instance_acc1 + test_instance_acc2)/2
            class_acc = (class_acc1 + class_acc2)/2

            if (test_instance_acc >= best_instance_acc):
                best_instance_acc = test_instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('FT-Test Instance Average Accuracy: %f, Class Average Accuracy: %f'% (test_instance_acc, class_acc))
            log_string('FT-Best Instance Average Accuracy: %f, Class Average Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                log_string('best_epoch %s' % str(best_epoch))
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

        logger_loss.append([train_loss, train_loss1, train_loss2, val_loss, val_loss1, val_loss2])
        logger_acc.append([train_instance_acc_ave, train_instance_acc, train_instance_acc2, test_instance_acc, test_instance_acc1, test_instance_acc2, signacc])

        time_end = datetime.datetime.now()
        time_span_str = str((time_end - time_start).seconds)
        log_string('Epoch time : %s S' % (time_span_str))

        time_end = datetime.datetime.now()
        time_span_str = str((time_end - time_start).seconds)
        log_string('Epoch time : %s S' % (time_span_str))

    logger_loss.close()
    logger_loss.plot()
    savefig(os.path.join(log_dir, 'log_loss.eps'))
    logger_acc.close()
    logger_acc.plot()
    savefig(os.path.join(log_dir, 'log_acc.eps'))

    log_string('best_epoch %s' % str(best_epoch))

    logger.info('End of fine-turning...')


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    main()
