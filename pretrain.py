import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
#from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm
import copy

import sys
import os
import glob
import argparse
import time
#from datetime import datetime
import random
import math

import robustbench as rb
import data
from autopgd_train import apgd_train, train_clean
import utils
from model_zoo.fast_models import PreActResNet18
import other_utils
import eval as utils_eval


eps_dict = {'cifar10': {'Linf': 8. / 255., 'L2': 0.5, 'L1': 12.},
    'imagenet': {'Linf': 4. / 255., 'L2': 2., 'L1': 255.}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Wong2020Fast')
    #parser.add_argument('--eps', type=float, default=8/255)
    #parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size_eval', type=int, default=100, help='batch size for evaluation')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--data_dir', type=str, default='/share/datasets/cifar10', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--save_dir', type=str, default='./trained_models')
    #parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--norm_idx', type=int, default=0, help='which one to pretrain: 0 - Linf, 1 - L1, 2 - L2')
    #parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--lr-schedule', default='piecewise-ft')
    parser.add_argument('--lr-max', default=.01, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    #parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=-1, help='if -1 no evaluation during training')
    parser.add_argument('--act', type=str, default='softplus1')
    parser.add_argument('--finetune_model', action='store_true')
    parser.add_argument('--l_norms', type=str, default='Linf L1 L2', help='norms to use in adversarial training')
    parser.add_argument('--attack', type=str, default='apgd')
    #parser.add_argument('--pgd_iter', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4) #5e-4)
    parser.add_argument('--l_eps', type=str, help='epsilon values for adversarial training wrt each norm')
    parser.add_argument('--notes_run', type=str, help='appends a comment to the run name')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--l_iters', type=str, help='iterations for each norms in adversarial training (possibly different)')
    #parser.add_argument('--epoch_switch', type=int)
    #parser.add_argument('--save_min', type=int, default=0)
    parser.add_argument('--save_optim', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    #parser.add_argument('--no_wd_bn', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--at_iter', type=int, help='iteration in adversarial training (used for all norms)')
    parser.add_argument('--n_ex_eval', type=int, default=200)
    parser.add_argument('--n_ex_final', type=int, default=1000)
    parser.add_argument('--n_examples', type=int, default=0)
    parser.add_argument('--final_eval', action='store_true', help='run long evaluation after training')
    parser.add_argument('--fname', type=str, help='store file name')
    parser.add_argument('--pretraining', action='store_true', help='whether do natural pretraining before advesarial training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # logging and saving tools
    # utils.get_runname(args)
    print(args.fname)
    other_utils.makedir('{}/{}'.format(args.save_dir, args.fname)) #args.save_dir
    files = glob.glob('{}/{}/*'.format(args.save_dir, args.fname))
    for f in files:
        os.remove(f)
    args.all_norms = ['Linf', 'L2', 'L1']
    args.all_epss = [eps_dict[args.dataset][c] for c in args.all_norms]
    stats = utils.stats_dict(args)
    logger = other_utils.Logger('{}/{}/log_train.txt'.format(args.save_dir,
        args.fname))
    log_eval_path = '{}/{}/log_eval_final.txt'.format(args.save_dir, args.fname)
    
    # fix seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    
    # load data
    if args.dataset == 'cifar10':
        train_loader, _ = data.load_cifar10_train(args, only_train=True)
        
        # non augmented images for statistics
        x_train_eval, y_train_eval = data.load_cifar10(args.n_ex_eval,
            args.data_dir, training_set=True, device='cuda')
        x_test_eval, y_test_eval = data.load_cifar10(args.n_ex_eval,
            args.data_dir, device='cuda') #training_set=True
        
        args.n_cls = 10
    else:
        raise NotImplemented
    print('data loaded on {}'.format(x_test_eval.device))
    
    # load model
    if not args.finetune_model:
        assert args.dataset == 'cifar10'
        #from model_zoo.fast_models import PreActResNet18
        model = PreActResNet18(10, activation=args.act).cuda()
        model.eval()
    elif args.model_name.startswith('RB'):
        #raise NotImplemented
        model = rb.utils.load_model(args.model_name.split('_')[1], model_dir=args.model_dir,
            dataset=args.dataset, threat_model=args.model_name.split('_')[2])
        model.cuda()
        model.eval()
        print('{} ({}) loaded'.format(*args.model_name.split('_')[1:]))
    elif args.model_name.startswith('pretr'):
        model = utils.load_pretrained_models(args.model_name)
        model.cuda()
        model.eval()
        print('pretrained model loaded')

    clean_acc = rb.utils.clean_accuracy(model, x_test_eval, y_test_eval)
    print('initial clean accuracy: {:.2%}'.format(clean_acc))

    # set loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = optim.SGD(get_params_no_decay(args, model), lr=1., momentum=0.9,
        weight_decay=args.weight_decay)

    model_nat = copy.deepcopy(model).cuda()
    optimizer_nat = optim.SGD(get_params_no_decay(args, model_nat), lr=1., momentum=0.9,
        weight_decay=args.weight_decay)
    
    # get lr scheduler
    lr_schedule = utils.get_lr_schedule(args)

    # set norms, eps and iters for training
    args.l_norms = args.l_norms.split(' ')
    if args.l_eps is None:
        args.l_eps = [eps_dict[args.dataset][c] for c in args.l_norms]
    else:
        norm, size = args.l_eps.split('_')[0], float(args.l_eps.split('_')[1])
        eps_dict[args.dataset][norm] = size
        args.l_eps = [eps_dict[args.dataset][c] for c in args.l_norms]
        
    if not args.l_iters is None:
        args.l_iters = [int(c) for c in args.l_iters.split(' ')]
    else:
        args.l_iters = [args.at_iter + 0 for _ in args.l_norms]
    print('[train] ' + ', '.join(['{} eps={:.5f} iters={}'.format(
        args.l_norms[c], args.l_eps[c], args.l_iters[c]) for c in range(len(
        args.l_norms))]))

    # set eps for evaluation
    for i, norm in enumerate(args.l_norms):
        idx = args.all_norms.index(norm)
        args.all_epss[idx] = args.l_eps[i] + 0.
    print('[eval] ' + ', '.join(['{} eps={:.5f}'.format(args.all_norms[c],
        args.all_epss[c]) for c in range(len(args.all_norms))]))
    model_old = copy.deepcopy(model).cuda()

    # pretraining on D_nat
    if args.pretraining:
        for i in range(40):
            model_nat, _ = train_clean(args, i, model_nat, train_loader, optimizer_nat, lr_schedule)
    
        model.load_state_dict(model_nat.state_dict())
    
    # training loop
    for epoch in range(0, args.epochs):  # loop over the dataset multiple times
        startt = time.time()
        model_old.load_state_dict(model.state_dict())
        if True:
            model.train()
            running_loss = 0.0
            running_acc = 0.
            running_acc_ep = 0.
            if epoch == 0: #epoch_init
                acc_norms = [[0., 0.] for _ in range(len(args.l_norms))]
            loss_norms = {k: [0., 0.] for k in args.l_norms}

            time_prev = time.time()
            with tqdm(train_loader, unit="batch") as tepoch:
                for i, (x_loader, y_loader) in enumerate(tepoch):
                    x, y = x_loader.cuda(), y_loader.cuda()

                    # update lr
                    lr = lr_schedule(epoch + (i + 1) / len(train_loader))
                    optimizer.param_groups[0].update(lr=lr)

                    if not args.attack == '':
                        model.eval()
                        
                        # sample which norm to use for the current batch
                        # if all([val[1] > 0 for val in acc_norms]):
                        #     ps = [val[0] / val[1] for val in acc_norms]
                        # else:
                        #     ps = [.5] * len(acc_norms)
                        # ps = [1. - val for val in ps]
                        # norm_curr = random.choices(range(len(ps)), weights=ps)[0]

                        # compute training points
                        if args.attack == 'apgd':
                            x_tr, acc_tr, _, _, _ = apgd_train(model, x, y, norm=args.l_norms[args.norm_idx],
                                eps=args.l_eps[args.norm_idx], n_iter=args.l_iters[args.norm_idx])
                            y_tr = y.clone()
                        else:
                            raise NotImplemented
                        
                        # update statistics
                        # acc_norms[norm_curr][0] += acc_tr.sum()
                        # acc_norms[norm_curr][1] += x.shape[0]
                        
                        model.train()
                    else:
                        # standard training
                        x_tr = x.clone()
                        y_tr = y.clone()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    if args.loss in ['ce']:
                        outputs = model(x_tr)
                        loss = criterion(outputs, y_tr)
                    loss.backward()
                    optimizer.step()

                    # collect stats
                    running_loss += loss.item() #w_tr
                    #running_acc += (outputs.max(dim=-1)[1] == y_tr).cpu().float().sum().item()
                    running_acc_ep += (outputs.max(dim=-1)[1] == y_tr).cpu().float().sum().item()
                    
                    # track loss for each norm
                    # if not args.attack is None:
                    #     loss_norms[args.l_norms[norm_curr]][0] += loss.item()
                    #     loss_norms[args.l_norms[norm_curr]][1] += 1

                    # logging
                    time_iter = time.time() - time_prev
                    time_prev = time.time()
                    time_cum = time.time() - startt
                    if len(args.l_norms) > 0:
                        other_stats = ' [indiv] ' + ', '.join(['{} {:.5f}'.format(k,
                            v[0] / max(1, v[1])) for k, v in loss_norms.items()])
                    else:
                        other_stats = ''

                    tepoch.set_postfix({'loss': running_loss / (i + 1), 'acc': running_acc_ep / (i + 1) / args.batch_size})

        model.eval()
        
        # training stats
        stats['loss_train_dets']['clean'][epoch] = running_loss / len(train_loader)

        str_to_log = '[epoch] {} [time] {:.1f} s [train] loss {:.5f}'.format(
                epoch + 1, time.time() - startt, stats['loss_train_dets']['clean'][epoch]) #stats['rob_acc_train_dets']['clean'][epoch]
        
        # compute robustness stats (apgd with 100 iterations)
        if (epoch + 1) % args.eval_freq == 0 and args.eval_freq > -1:
            # training points
            acc_train = utils_eval.eval_norms_fast(model, x_train_eval, y_train_eval,
                args.all_norms, args.all_epss, n_iter=100, n_cls=args.n_cls)
            # test points
            acc_test = utils_eval.eval_norms_fast(model, x_test_eval, y_test_eval,
                args.all_norms, args.all_epss, n_iter=100, n_cls=args.n_cls)
            str_test, str_train = '', ''
            for norm in args.all_norms + ['clean', 'union']:
                stats['rob_acc_test_dets'][norm][epoch] = acc_test[norm]
                stats['rob_acc_train_dets'][norm][epoch] = acc_train[norm]
                str_test += ' {} {:.1%}'.format(norm, acc_test[norm])
                str_train += ' {} {:.1%}'.format(norm, acc_train[norm])
            str_to_log += '\n'
            str_to_log += '[eval train]{} [eval test]{}'.format(str_train, str_test)
        
        # saving
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            curr_dict = model.state_dict()
            if args.save_optim:
                curr_dict = {'state_dict': model.state_dict(), 'optim': optimizer.state_dict()}
            torch.save(curr_dict, '{}/{}/ep_{}.pth'.format(
                    args.save_dir, args.fname, epoch + 1))
            torch.save(stats, '{}/{}/metrics.pth'.format(args.save_dir, args.fname))

        logger.log(str_to_log)

    # run long eval
    if args.final_eval:
        x, y = data.load_cifar10(args.n_ex_final, data_dir=args.data_dir, device='cpu')
        l_x_adv, stats['final_acc_dets'] = utils_eval.eval_norms(model, x, y,
            l_norms=args.all_norms, l_epss=args.all_epss,
            bs=args.batch_size_eval, log_path=log_eval_path) #model, args=args
        torch.save(stats, '{}/{}/metrics.pth'.format(args.save_dir, args.fname))
        for norm, eps, v in zip(args.l_norms, args.l_eps, l_x_adv):
            torch.save(v,  '{}/{}/eval_{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, args.fname, 'final', norm, args.n_ex_final, eps))



if __name__ == '__main__':
    main()

