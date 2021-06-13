import os 
import sys
import argparse 
from utils_cm import str2bool
from pprint import pprint

WP = os.path.dirname(os.path.realpath('__file__')) +'/'
print(WP)

parser = argparse.ArgumentParser()
# TRAINING SETTING 
parser.add_argument("--bs", 		type=int, 		default=128, 		help="batch_size")
parser.add_argument("--ds", 		type=str, 		default='mnist', 	help="dataset")
parser.add_argument("--model", 		type=str, 		default='cnn',      help="model type")
parser.add_argument("--activ", 	    type=str, 		default='relu', 	help="activation function")
parser.add_argument("--epochs", 	type=int, 		default=100, 	 	help="number epoch for training")
parser.add_argument("--no_cuda",    type=str2bool, 	default=False, 		help="Use cuda or not")
parser.add_argument("--inf", 		type=str, 		default='none', 	help="additional info")
parser.add_argument("--save_freq", 	type=int, 		default=10, 		help="batch_size")

# ADVERSARY TRAINING SETTING 
parser.add_argument("--defense", 	type=str, 		default='none',     help="defense method")
parser.add_argument("--loss_type", 	type=str, 	    default='ce', 		help="loss type, ce or kl")
parser.add_argument("--order", 	    type=str, 	    default='inf', 		help="distance order")
parser.add_argument("--random_init", type=str2bool, 	default=True, 		help="Random init starting point")
parser.add_argument("--epsilon", 	type=float, 	default=0.3, 		help="epsilon for attack")
parser.add_argument("--step_size", 	type=float, 	default=-1., 		help="step_size for attack")

# EVAL SETTING 
parser.add_argument("--attack_type",        type=str, 	    default='pgd_attack',   help="attack method")
parser.add_argument("--eval_attack_type",   type=str, 	    default='pgd_attack',   help="attack method for eval")
parser.add_argument("--eval_loss_type", 	type=str,       default='ce',           help="loss type, ce or kl")
parser.add_argument("--eval_random_init",   type=str2bool,  default=True, 	        help="Random init starting point")
parser.add_argument("--eval_projecting",    type=str2bool, 	default=True, 	        help="Projecting to perturbation bound or not")
parser.add_argument("--eval_auto",          type=str2bool, 	default=True, 	        help="Evaluation with Auto-Attack")
parser.add_argument("--eval_bb",            type=str2bool, 	default=True, 	        help="Evaluation with B&B Attack")
parser.add_argument("--eval_multi",         type=str2bool, 	default=False, 	        help="PGD Attack with multiple epsilons")
parser.add_argument("--eval_multi_l2",      type=str2bool, 	default=False, 	        help="PGD Attack with multiple epsilons with L2 norm")
parser.add_argument("--eval_multi_auto",    type=str2bool, 	default=False, 	        help="Evaluation with Auto-Attack")
parser.add_argument("--eval_multi_bb",      type=str2bool, 	default=False, 	        help="Evaluation with B&B Attack")
parser.add_argument("--eval_scan",          type=str2bool, 	default=False, 	        help="Evaluation will all saved checkpoints in folder")
parser.add_argument("--eval_epoch",         type=int, 	    default=-1, 	        help="Evaluation with specific checkpoint")
parser.add_argument("--eval_best",          type=str2bool, 	default=False, 	        help="Evaluate the best checkpoint or the last checkpoint")

# METHOD PARAMS
parser.add_argument("--projecting", type=str2bool, 	    default=True, 	    help="Projecting to perturbation bound or not")
parser.add_argument("--distype", 	type=str,           default='smooth',   help="distance type, 'none', 'smooth', 'shift' ")
parser.add_argument("--lccomw", 	type=float, 	    default=1.0, 		help="local compactness param")
parser.add_argument("--lcsmtw", 	type=float, 	    default=1.0, 		help="local smoothness param")
parser.add_argument("--gbcomw", 	type=float, 	    default=1.0, 		help="global compactness param")
parser.add_argument("--gbsmtw", 	type=float, 	    default=1.0, 		help="global smoothness param")
parser.add_argument("--confw", 	    type=float, 	    default=1.0, 		help="confidence loss param")


# TRADES
parser.add_argument("--trades_beta", 	type=float, 	default=1.0, 	    help="parameter for TRADES")

args = parser.parse_args()

pprint(vars(args))

basesetup = [
    ('ds={}',		args.ds), 
    ('model={}',	args.model), 
    ('activ={}',	args.activ), 
    ('bs={}', 		args.bs),
]	

setup = [
    ('defense={}',	args.defense), 
    ('beta={}',	    args.trades_beta), 
    ('lccomw={}',	args.lccomw), 
    ('lcsmtw={}',	args.lcsmtw), 
    ('gbcomw={}',	args.gbcomw), 
    ('gbsmtw={}',	args.gbsmtw), 
    ('confw={}',	args.confw), 
    ('inf={}', 		args.inf), 
]

basedir = '_'.join([t.format(v) for (t, v) in basesetup])
modeldir = '_'.join([t.format(v) for (t, v) in setup])