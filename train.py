from data_multi import Bandit_multi
from learner_diag import NeuralUCBDiag
import numpy as np
import argparse
import pickle
import os
import time
import torch


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    parser = argparse.ArgumentParser(description='NeuralUCB')



    parser.add_argument('--size', default=15000, type=int, help='bandit size')
    parser.add_argument('--dataset', default='mnist', metavar='DATASET')
    parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not')
    parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')
    parser.add_argument('--hidden', type=int, default=100, help='network hidden size')



    args = parser.parse_args()
    use_seed = None if args.seed == 0 else args.seed
    b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
    bandit_info = '{}'.format(args.dataset)
    l = NeuralUCBDiag(b.dim, args.lamdba, args.nu, args.hidden)
    ucb_info = '_{:.3e}_{:.3e}_{}'.format(args.lamdba, args.nu, args.hidden)


    regrets = []
    summ = 0
    for t in range(min(args.size, b.size)):
        context, rwd = b.step()
        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        if t<2000:
            loss = l.train(context[arm_select], r)
        else:
            if t%100 == 0:
                loss = l.train(context[arm_select], r)
        regrets.append(summ)
        if t % 100 == 0:
            print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss, nrm, sig, ave_rwd))

    path = '{}_{}_{}'.format(bandit_info, ucb_info, time.time())
    fr = open(path,'w')
    for i in regrets:
        fr.write(str(i))
        fr.write("\n")
    fr.close()
