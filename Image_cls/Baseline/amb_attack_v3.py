import argparse
from pprint import pprint

from experiments.classification_private import ClassificationPrivateExperiment
import shutil

def main():
    parser = argparse.ArgumentParser()

    # passport argument
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--key-type', choices=['random', 'image', 'shuffle'], default='random',
                        help='passport key type (default: shuffle)')
    parser.add_argument('-s', '--sign-loss', type=float, default=0.1,
                        help='sign loss to avoid scale not trainable (default: 0.1)')
    parser.add_argument('--use-trigger-as-passport', action='store_true', default=False,
                        help='use trigger data as passport')

    parser.add_argument('--train-passport', action='store_true', default=False,
                        help='train passport')
    parser.add_argument('-tb', '--train-backdoor', action='store_true', default=True,
                        help='train backdoor')
    parser.add_argument('--train-private', action='store_true', default=True,
                        help='train private')  # always true

    # paths
    parser.add_argument('--pretrained-path',
                        help='load pretrained path')
    parser.add_argument('--lr-config', default='lr_configs/default.json',
                        help='lr config json file')


    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')

    # transfer learning
    parser.add_argument('-tf', '--transfer-learning', action='store_true', default=False,
                        help='turn on transfer learning')
    parser.add_argument('--tl-dataset', default='cifar100', choices=['cifar10',
                                                                     'cifar100',
                                                                     'caltech-101',
                                                                     'caltech-256'],
                        help='transfer learning dataset (default: cifar100)')
    parser.add_argument('--tl-scheme', default='rtal', choices=['rtal',
                                                                'ftal'],
                        help='transfer learning scheme (default: rtal)')

    parser.add_argument('--attack', action='store_true', default=True,  help='attack the pretrained model')  # always true

    parser.add_argument('-n', '--norm-type', default='gn', choices=['bn', 'gn', 'in', 'sbn', 'none'],
                        help='norm type (default: bn)')
    parser.add_argument('-t', '--tag', default='all', help='tag')
    parser.add_argument('--arch', default='resnet', choices=['alexnet', 'resnet'],
                        help='architecture (default: alexnet)')
    parser.add_argument('--passport-config', default='passport_configs/resnet18_passport.json',
                        help='should be same json file as arch')

    parser.add_argument('-d', '--dataset', default='cifar100', choices=['cifar10',  'cifar100'],
                        help='training dataset (default: cifar10)')

    parser.add_argument('--type', default='fake2-10', help='fake key type, fake2, fake3_100S')
    parser.add_argument('--flipperc', default=1, type=float, help='flip percentange 0~1')
    args = parser.parse_args()

    pprint(vars(args))

    if 'fake2-' in args.type  :
        args.flipperc = 0
        print('No Flip')
    elif  'fake3-' in args.type:
        args.flipperc = 0
        print('No Flip')

    print("Filp %d percentage sign"%(args.flipperc * 100))

    exp = ClassificationPrivateExperiment(vars(args))
    task_name = exp.logdir.split('/')[-2]
    logdir = '/data-x/g12/zhangjie/DeepIPR/baseline/passport_attack/' + task_name + '/' + args.type
    exp.fake_attack()
    shutil.copy('amb_attack_v3.py', str(logdir) + "/abm_attack_v3.py")

    print('Ambiguity attack done at', logdir)


if __name__ == '__main__':
    main()
