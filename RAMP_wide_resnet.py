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
import argparse
import time
#from datetime import datetime
import random
import math
import glob

import robustbench as rb
import data
from autopgd_train import apgd_train, train_clean
from utils import gp, get_params_no_decay
import utils
from model_zoo.fast_models import PreActResNet18
from model_zoo.wide_resnet import WideResNet
import other_utils
import eval as utils_eval


eps_dict = {'cifar10': {'Linf': 8. / 255., 'L2': .5, 'L1': 12.},
    'imagenet': {'Linf': 4. / 255., 'L2': 2., 'L1': 255.}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Wong2020Fast')
    #parser.add_argument('--eps', type=float, default=8/255)
    #parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size_eval', type=int, default=100, help='batch size for evaluation')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--data_dir', type=str, default='../datasets/cifar10', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--save_dir', type=str, default='./trained_models')
    #parser.add_argument('--norm', type=str, default='Linf')
    #parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--lr-schedule', default='piecewise-ft')
    parser.add_argument('--lr-max', default=.01, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    #parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=3, help='if -1 no evaluation during training')
    parser.add_argument('--act', type=str, default='softplus1')
    parser.add_argument('--finetune_model', action='store_true')
    parser.add_argument('--l_norms', type=str, default='Linf L1', help='norms to use in adversarial training')
    parser.add_argument('--attack', type=str, default='apgd')
    #parser.add_argument('--pgd_iter', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--l_eps', type=str, help='epsilon values for adversarial training wrt each norm')
    parser.add_argument('--notes_run', type=str, help='appends a comment to the run name')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--l_iters', type=str, help='iterations for each norms in adversarial training (possibly different)')
    #parser.add_argument('--epoch_switch', type=int)
    #parser.add_argument('--save_min', type=int, default=0)
    parser.add_argument('--save_optim', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fname', type=str, help='store file name')
    #parser.add_argument('--no_wd_bn', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--n_examples', type=int, default=0)
    parser.add_argument('--at_iter', type=int, help='iteration in adversarial training (used for all norms)')
    parser.add_argument('--n_ex_eval', type=int, default=200)
    parser.add_argument('--n_ex_final', type=int, default=10000)
    parser.add_argument('--final_eval', action='store_true', help='run long evaluation after training')

    # parameters related to RAMP
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--lbd', type=float, default=1.5)
    parser.add_argument('--pretraining', action='store_true')
    parser.add_argument('--max', action='store_true') # whether to use max strategy for L1 Linf perturb
    parser.add_argument('--wide', action='store_true')

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
    elif args.dataset == 'imagenet':
        train_loader, _ = data.load_imagenet_train(args)
        
        # non augmented images for statistics
        x_train_eval, y_train_eval = data.load_imagenet(args.n_ex_eval)
        x_test_eval, y_test_eval = data.load_imagenet(args.n_ex_eval) #training_set=True
        
        args.n_cls = 1000
    else:
        raise NotImplemented
    print('data loaded on {}'.format(x_test_eval.device))
    
    # load model
    if not args.finetune_model:
        assert args.dataset == 'cifar10'
        #from model_zoo.fast_models import PreActResNet18
        if args.wide:
            model = WideResNet().cuda()
        else:
            model = PreActResNet18(10, activation=args.act).cuda()
        model.eval()
    elif args.model_name.startswith('RB'):
        #raise NotImplemented
        model = rb.utils.load_model('_'.join(args.model_name.split('_')[1:]), model_dir=args.model_dir,
            dataset=args.dataset, threat_model='Linf')
        model.cuda()
        model.eval()
        print('{} ({}) loaded'.format(*args.model_name))
    elif args.model_name.startswith('pretr'):
        model = utils.load_pretrained_models(args.model_name)
        model.cuda()
        model.eval()
        print('pretrained model loaded')
    else:
        model = PreActResNet18(10, activation=args.act)
        ckpt = torch.load(args.model_name)
        model.load_state_dict(ckpt)
        model.cuda()
        model.eval()

    # clean_acc = rb.utils.clean_accuracy(model, x_test_eval, y_test_eval)
    # print('initial clean accuracy: {:.2%}'.format(clean_acc))

    # set loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = optim.SGD(get_params_no_decay(args, model), lr=1., momentum=0.9,
        weight_decay=args.weight_decay)
    
    # initialize models for D_nat and D_inf
    model_nat = copy.deepcopy(model)
    model_inf = copy.deepcopy(model)
    optimizer_nat = optim.SGD(get_params_no_decay(args, model_nat), lr=1., momentum=0.9,
        weight_decay=args.weight_decay)
    optimizer_inf = optim.SGD(get_params_no_decay(args, model_inf), lr=1., momentum=0.9,
        weight_decay=args.weight_decay)

    # get lr scheduler
    lr_schedule = utils.get_lr_schedule(args)

    # set norms, eps and iters for training
    args.l_norms = args.l_norms.split(' ')
    if args.l_eps is None:
        args.l_eps = [eps_dict[args.dataset][c] for c in args.l_norms]
    else:
        args.l_eps = [float(c) for c in args.l_eps.split(' ')]
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
    
    cur_norm_source = 0 # initial source - Linf
    cur_norm_target = 1 # initial target - L1
    iteration = 0

    # pretraining on D_nat
    if args.pretraining:
        for i in range(40):
            model_nat, _ = train_clean(args, i, model_nat, train_loader, optimizer_nat, lr_schedule)
    
        model.load_state_dict(model_nat.state_dict())


    
    for epoch in range(0, args.epochs):  # loop over the dataset multiple times
        model_old = copy.deepcopy(model)
        startt = time.time()
        if True:
            
            # training the target domain
            model.train()
            model_inf.load_state_dict(model.state_dict())
            model_nat.load_state_dict(model.state_dict())
            

            # train the natural domain
            if args.gp:
                model_nat, _ = train_clean(args, epoch, model_nat, train_loader, optimizer_nat, lr_schedule)
            
            # train the Lp domain (target domain)
            with tqdm(train_loader, unit="batch") as tepoch:
                running_loss_t = 0.0
                running_acc_ep_s = 0.
                running_acc_ep_t = 0.
                for i, (x_loader, y_loader) in enumerate(tepoch):
                    x, y = x_loader.cuda(), y_loader.cuda()

                    # update lr
                    lr = lr_schedule(epoch + (i + 1) / len(train_loader))
                    optimizer.param_groups[0].update(lr=lr)

                    model.eval()

                    # compute training points
                    x_tr_s, _, _, loss_best_s, _ = apgd_train(model, x, y, norm=args.l_norms[cur_norm_source],
                                eps=args.l_eps[cur_norm_source], n_iter=args.l_iters[cur_norm_source], is_train=True)                  

                    x_tr_t, _, _, loss_best_t, _ = apgd_train(model, x, y, norm=args.l_norms[cur_norm_target],
                                eps=args.l_eps[cur_norm_target], n_iter=args.l_iters[cur_norm_target], is_train=True)
                    
                    
                    y_tr = y.long().clone()

                    if args.max:
                        tensor_list = [loss_best_t, loss_best_s]
                        delta_list = [x_tr_t.view(len(y),1,-1), x_tr_s.view(len(y),1,-1)]
                        loss_arr = torch.stack(tuple(tensor_list))
                        delta_arr = torch.stack(tuple(delta_list))
                        max_loss = loss_arr.max(dim = 0)
                        x_tr_best = delta_arr[max_loss[1], torch.arange(len(y)), 0]
                        x_tr_best = x_tr_best.view(len(y), x_tr_s.shape[1], x_tr_s.shape[2], x_tr_s.shape[3])
                    
                        
                    model.train()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    if args.loss in ['ce']:
                        outputs_t = model(x_tr_t)
                        outputs_s = model(x_tr_s)

                        acc_ep_t = (outputs_t.max(dim=-1)[1] == y_tr).cpu().float().sum().item()
                        running_acc_ep_t += acc_ep_t
                        acc_ep_s = (outputs_s.max(dim=-1)[1] == y_tr).cpu().float().sum().item()
                        running_acc_ep_s += acc_ep_s
                                               

                        loss_kl = 0

                        if args.kl or args.mse:
                            if args.kl:
                                criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                            else:
                                criterion_kl = nn.MSELoss()

                            source_correct_indices =  (outputs_s.max(dim=-1)[1] == y_tr).detach()
                            
                            # kl

                            selected_kl = source_correct_indices
                            source_sel_kl, target_sel_kl = outputs_s[selected_kl], outputs_t[selected_kl]

                            if len(source_sel_kl) > 0:
                                if args.kl:
                                    loss_kl = criterion_kl(F.log_softmax(target_sel_kl+1e-12, dim=1), F.softmax(source_sel_kl, dim=1)) / selected_kl.sum() # (selected_kl.sum() / len(y))
                                else:
                                    loss_kl = criterion_kl(target_sel_kl, source_sel_kl)

                            # best
                            if args.max:
                                loss_natural = F.cross_entropy(model(x), y)
                                loss_robust = (1.0 / len(y)) * criterion_kl(F.log_softmax(model(x_tr_best), dim=1),
                                                                                F.softmax(model(x), dim=1))
                                
                        
                        
                        loss = loss_natural + 6. * loss_robust + loss_kl * args.lbd

                    loss.backward()
                    optimizer.step()

                    # collect stats
                    running_loss_t += loss.item() #w_tr
                    
                        
                    tepoch.set_postfix({'loss': running_loss_t / (i+1), 'acc_s': running_acc_ep_s / (i + 1) / args.batch_size, 'acc_t': running_acc_ep_t / (i + 1) / args.batch_size})
            
            if args.gp:
                # model fusion using GP
                st = time.time()
                model_dict = gp(0.5, [model_nat.state_dict()], model_old.state_dict(), model.state_dict(), [1.0])
                print(time.time() - st)
                model_nat.load_state_dict(model_dict)
                model.load_state_dict(model_dict)
            

        model.eval()
        
        # training stats
        stats['loss_train_dets']['clean'][epoch] = running_loss_t / len(train_loader) 

        str_to_log = '[epoch] {} [time] {:.1f} s [train] loss {:.5f}'.format(
                epoch + 1, time.time() - startt, stats['loss_train_dets']['clean'][epoch]) 
        
        # compute robustness stats (apgd with 100 iterations)
        if (epoch + 1) % args.eval_freq == 0 and args.eval_freq > -1:
            model.eval()
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
            str_to_log += '[eval train]{} [eval test]{}'.format(str_train, str_test)

    
        # saving
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            curr_dict = model.state_dict()
            if args.save_optim:
                curr_dict = {'state_dict': model.state_dict(), 'optim': optimizer.state_dict()}
            torch.save(curr_dict, '{}/{}/ep_{}_{}.pth'.format(
                    args.save_dir, args.fname, epoch + 1, iteration))
            torch.save(stats, '{}/{}/metrics.pth'.format(args.save_dir, args.fname))

            

        logger.log(str_to_log)

            


    # run long eval
    if args.final_eval:
        if args.dataset == 'cifar10':
            x, y = data.load_cifar10(args.n_ex_final, data_dir=args.data_dir, device='cpu')
            l_x_adv, stats['final_acc_dets'] = utils_eval.eval_norms(model, x, y,
            l_norms=args.all_norms, l_epss=args.all_epss,
            bs=args.batch_size_eval, log_path=log_eval_path)
        else:
            x, y = data.load_imagenet(args.n_ex_final, device='cpu')
            l_x_adv, stats['final_acc_dets'] = utils_eval.eval_norms(model, x, y,
            l_norms=args.all_norms, l_epss=args.all_epss,
            bs=args.batch_size_eval, log_path=log_eval_path, n_cls=1000) #model, args=args
        torch.save(stats, '{}/{}/metrics_{}.pth'.format(args.save_dir, args.fname, iteration))
        for norm, eps, v in zip(args.l_norms, args.l_eps, l_x_adv):
            torch.save(v,  '{}/{}/eval_{}_{}_1_{}_eps_{:.5f}_{}.pth'.format(
                args.save_dir, args.fname, 'final', norm, args.n_ex_final, eps, iteration))



if __name__ == '__main__':
    main()

