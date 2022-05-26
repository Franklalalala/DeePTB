#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io, os, sys

import argparse
from dpnegf.NNSKTB import deepTB, dpTBtrain, dpTBtest
from dpnegf.NNEGF import deepnegf
from dpnegf.band import band_plot

try:
    # Python 3, open as binary, then wrap in a TextIOWrapper
    unbuffered = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
except TypeError:
    # Python 2
    unbuffered = os.fdopen(sys.stdout.fileno(), 'w', 0)
    
sys.stdout = unbuffered

__author__ = "Q. Gu, et al."
__copyright__ = "Q. Gu & AISI"
__status__ = "developing"

def main():
    parser = argparse.ArgumentParser(
        description="DeepNEGF: A deep learning package for quantum transport simulations with"
        " non-equilibrium Green's function (NEGF) approach. Type 'dpnegf subcommand -h' for instructions."
    ) 

    subparsers = parser.add_subparsers(title="valid subcommands", dest="command")

    paras_tbtrain = subparsers.add_parser("train", 
        help='train NN to get the TB hamiltonians with labeled data.')
    paras_tbtrain.add_argument("-r", "--restart", type=str, default = None,
        help='training the model initialed with checkpoint.')
    paras_tbtrain.add_argument('-i', '--input_file', type=str,
        default='input.json', help='json file for inputs, default input.json')

    paras_test = subparsers.add_parser("test", 
        help='test the trained NN. using the unseen labeled data.')
    paras_test.add_argument('-i', '--input_file', type=str,
        default='input.json', help='json file for inputs, default input.json')

    # paras_pred = subparsers.add_parser("predict", 
    #    help='using the trained NN to make prediction for unlabeled data.')
    # paras_test.add_argument('-i', '--input_file', type=str,
    #    default='input.json', help='json file for inputs, default inputnn.json')

    paras_plotband = subparsers.add_parser("band", 
        help='Perform NEGF simulations with NN-baed TB Hamiltonians.')
    paras_plotband.add_argument('-i', '--input_file', type=str,
        default='input.json', help='json file for inputs, default input.json')
    paras_plotband.add_argument('--struct', type=str, default=None, 
        help='the struct to calculate band.') 
    paras_plotband.add_argument('--format', type=str, default='vasp', 
        help='the format of struct to calculate band. default vasp.') 
    paras_plotband.add_argument('--nn_off', type=str, default=None, 
        help='turn off nn correction, only use sktb as Hamiltion, default None, means use NN corrected TB.')  

    
    paras_negf = subparsers.add_parser("negf", 
        help='Perform NEGF simulations with NN-baed TB Hamiltonians.')
    paras_negf.add_argument('-i', '--input_file', type=str,
        default='input.json', help='json file for inputs, default input.json')
    paras_negf.add_argument('-s', '--struct', type=str, default='struct.xyz',
        help='struct file name default struct.xyz')
    paras_negf.add_argument('-fmt', '--format', type=str, default='xyz',
        help='struct file format default xyz')
    paras_negf.add_argument('--use_ase', type=str, default=None, 
        help='call ase transport API.')  
    paras_negf.add_argument('--nn_off', type=str, default=None, 
        help='turn off nn correction, only use sktb as Hamiltion, default None, means use NN corrected TB.')  

    args = parser.parse_args()

    if args.command=='train':
        dpTBtrain(args)
    elif args.command=='test':
        dpTBtest(args)
    elif args.command=='negf':
        deepnegf(args)
    elif args.command=='band':
        band_plot(args)
        
    
if __name__ == "__main__":
    main()

