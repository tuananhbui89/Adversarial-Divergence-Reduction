import os 
import sys
import argparse 
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument("--ds", 		type=str, 		default='cifar10', 	help="dataset")
parser.add_argument("--model", 		type=str, 		default='cnn', 		help="model type")
parser.add_argument("--defense", 	type=str, 		default='none', 	help="Defense method")
parser.add_argument("--perturb", 	type=str, 		default='pgd', 		help="perturbation type")
parser.add_argument("--inf", 		type=str, 		default='none', 	help="additional info")

parser.add_argument("--bs", 		type=int, 		default=128, 		help="batch size")
parser.add_argument("--lccomw", 	type=float, 	default=0., 		help="weight of local compactness")
parser.add_argument("--confw", 		type=float, 	default=0., 		help="weight of confidence loss")
parser.add_argument("--vatw", 		type=float, 	default=0., 		help="weight of vat loss")
parser.add_argument("--gbcomw", 	type=float, 	default=0., 		help="weight of global compactness")
parser.add_argument("--gbsmtw", 	type=float, 	default=0., 		help="weight of global smoothness")
parser.add_argument("--wdc", 		type=float, 	default=0.0002, 	help="weight of weight decay")
parser.add_argument("--ord", 		type=int, 		default=1, 			help="latent distance order")

parser.add_argument("--atk_eps", 	type=float, 	default=8., 		help="attack epsilon")
parser.add_argument("--atk_steps", 	type=int, 		default=20, 		help="attack number of step")
args = parser.parse_args()

pprint(vars(args))

setup = [
	('ds={}',		args.ds), 
	('model={}',	args.model), 
	('defense={}', 	args.defense),
	('perturb={}', 	args.perturb),
	('bs={}', 		args.bs),
	('ord={}', 		args.ord),
	('lccomw={}', 	args.lccomw), 
	('confw={}', 	args.confw), 
	('vatw={}', 	args.vatw), 
	('gbcomw={}', 	args.gbcomw),
	('gbsmtw={}', 	args.gbsmtw), 
	('wdc={}', 		args.wdc), 
	('inf={}', 		args.inf)
]
