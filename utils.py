from datetime import datetime
import torch
import math
import numpy as np

def get_runname(args):
    args.fname = 'model_{}'.format(str(datetime.now()))
    args.fname += ' {} lr={:.5f} {} ep={}{} attack={} fts={} seed={}'.format(
        args.dataset, #+ ' ' if args.dataset != 'cifar10' else ''
        args.lr_max, args.lr_schedule, args.epochs, ' wd={}'.format(
            args.weight_decay) if args.weight_decay != 5e-4 else '',
        args.attack, #' act={}'.format(args.topcl_act) if not args.finetune_model or args.fts_idx == 'rand' else ''
        args.model_name if args.finetune_model else 'rand', #args.fts_idx
        args.seed)
    args.fname += ' at={}'.format(args.l_norms)
    #args.l_norms = args.l_norms.split(' ')
    if not args.l_eps is None:
        args.fname += ' eps={}'.format(args.l_eps)
    else:
        args.fname += ' eps=default'
    args.fname += ' iter={}'.format(args.at_iter if args.l_iters is None else args.l_iters)


def stats_dict(args):
    stats = {#'rob_acc_test': torch.zeros([args.epochs]),
        #'clean_acc_test': torch.zeros([args.epochs]),
        #'rob_acc_train': torch.zeros([args.epochs]),
        #'loss_train': torch.zeros([args.epochs]),
        'rob_acc_test_dets': {},
        'rob_acc_train_dets': {},
        'loss_train_dets': {},
        'freq_in_at': {},
        }
    #
    for norm in args.all_norms + ['union', 'clean']:
        stats['rob_acc_test_dets'][norm] = torch.zeros([args.epochs])
        stats['rob_acc_train_dets'][norm] = torch.zeros([args.epochs])
        if not norm in ['union']:
            stats['loss_train_dets'][norm] = torch.zeros([args.epochs])
        if not norm in ['union', 'clean']:
            stats['freq_in_at'][norm] = torch.zeros([args.epochs])
    return stats


def load_pretrained_models(modelname):
    from model_zoo.fast_models import PreActResNet18
    # from model_zoo.resnet_madry import resnet50
    # from model_zoo.resnet import ResNet50
    model = PreActResNet18(10, activation='softplus1').cuda()
    # model = ResNet50().cuda()
    # print(model)
    ckpt = torch.load('./models/{}.pth'.format(modelname))
    # sd = torch.load('./models/{}.pth'.format(modelname))['model']
    # # sd = {k[len('module.attacker.model.'):]:v for k,v in sd.items()}
    # ckpt = {}
    # for k in list(sd.keys())[-320:]:
    #     ckpt[k[len('module.attacker.model.'):]] = sd[k]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def get_lr_schedule(args):
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
        # lr_schedule = lambda t: np.interp([t], [0, args.epochs], [0, args.lr_max])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'piecewise-ft':
        def lr_schedule(t):
            if t / args.epochs < 1. / 3.:
                return args.lr_max
            elif t / args.epochs < 2. / 3.:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'piecewise-imagenet':
        def lr_schedule(batch_id, batch_num):
            if batch_id / batch_num < 1. / 3.:
                return args.lr_max
            elif batch_id / batch_num < 2. / 3.:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule.startswith('static'):
        def lr_schedule(t):
            if t > 70:
                return args.lr_max / 10.
            return args.lr_max
    elif args.lr_schedule.startswith('piecewise'):
        w = [float(c) for c in args.lr_schedule.split('-')[1:]]
        def lr_schedule(t):
            c = 0
            while t / args.epochs > sum(w[:c + 1]) / sum(w):
                c += 1
            return args.lr_max / 10. ** c

    return lr_schedule


def get_accuracy_and_logits(model, x, y, batch_size=100, n_classes=10):
    logits = torch.zeros([y.shape[0], n_classes], device='cpu')
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)

    # print(x.shape[0], batch_size)


    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].cuda()
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].cuda()

            output = model(x_curr)
            logits[counter * batch_size:(counter + 1) * batch_size] += output.cpu()
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0], logits


def gp(b, local_models_dict, old_global_model_dict, target_model_dict, clients_size_frac):
    ret_dict = copy.deepcopy(old_global_model_dict)
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]): # and ('bn' not in key and 'bias' not in key):
            global_grad = target_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * cur_sim * local_grad

            ret_dict[key] = ret_dict[key] + (1-b) * global_grad
        else:
            # we did not want to change the bn stats
            ret_dict[key] = target_model_dict[key]
    return ret_dict

# get params without decay
def get_params_no_decay(args, model):
    if args.weight_decay > 0 and not args.finetune_model: #args.no_wd_bn
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.weight_decay},
                  {'params': no_decay, 'weight_decay': 0}]
        print('not using wd for bn layers')
    else:
        params = model.parameters()
    return params