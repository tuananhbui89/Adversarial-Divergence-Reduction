# Adversarial-Divergence-Reduction (ADR)

This implementation corresponds with the paper ["Improving Adversarial Robustness by Enforcing Local and Global Compactness"](https://arxiv.org/abs/2007.05123) which has been accepted to ECCV-2020. [paper](https://arxiv.org/abs/2007.05123)[slide](https://www.dropbox.com/s/m7kdbte0rxh0qra/FIT_presentation_Sep_20.pdf?dl=0)

## References 
This implementation is mainly based on the [Cifar10-Challenge](https://github.com/MadryLab/cifar10_challenge) from Madry lab.  

## Requirements 
- Python 3.x
- Tensorflow 1.15
- scikit-learn 0.23.2
- seaborn 0.9.0

## Running 
Just run the file `runloop_cifar10.py` for the default setting with sequential training and testing. Refer to the file `mysetting.py` for parameter setting.  
Custom ADV Training:

	python train_compact.py --defense=adv --ds=cifar10  --model=nonwide --perturb=pgd --lccomw=0.0 --confw=0.0 --vatw=0.0 --gbcomw=0.0

	python eval_compact.py --defense=adv --ds=cifar10  --model=nonwide --perturb=pgd --lccomw=0.0 --confw=0.0 --vatw=0.0 --gbcomw=0.0

	python eval_multi_targeted.py --defense=adv --ds=cifar10  --model=nonwide --perturb=pgd --lccomw=0.0 --confw=0.0 --vatw=0.0 --gbcomw=0.0

Custom ADR-ADV Training: 

	python train_compact.py --defense=adr_adv --ds=cifar10  --model=nonwide --perturb=pgd --lccomw=1.0 --confw=1.0 --vatw=1.0 --gbcomw=1.0

	python eval_compact.py --defense=adr_adv --ds=cifar10  --model=nonwide --perturb=pgd --lccomw=1.0 --confw=1.0 --vatw=1.0 --gbcomw=1.0
	
	python eval_multi_targeted.py --defense=adr_adv --ds=cifar10  --model=nonwide --perturb=pgd --lccomw=1.0 --confw=1.0 --vatw=1.0 --gbcomw=1.0

## Cite 

If you find this work useful for your research, please consider citing:

    @inproceedings{bui2020improving,
      title={Improving Adversarial Robustness by Enforcing Local and Global Compactness},
      author={Bui, Anh and Le, Trung and Zhao, He and Montague, Paul and deVel, Olivier and Abraham, Tamas and Phung, Dinh},
      booktitle={European Conference on Computer Vision (ECCV), 2020},
      year={2020}
    }

or 

	@inproceedings{DBLP:conf/eccv/BuiLZMDAP20,
	  author    = {Anh Bui and
	               Trung Le and
	               He Zhao and
	               Paul Montague and
	               Olivier Y. DeVel and
	               Tamas Abraham and
	               Dinh Q. Phung},
	  editor    = {Andrea Vedaldi and
	               Horst Bischof and
	               Thomas Brox and
	               Jan{-}Michael Frahm},
	  title     = {Improving Adversarial Robustness by Enforcing Local and Global Compactness},
	  booktitle = {Computer Vision - {ECCV} 2020 - 16th European Conference, Glasgow,
	               UK, August 23-28, 2020, Proceedings, Part {XXVII}},
	  series    = {Lecture Notes in Computer Science},
	  volume    = {12372},
	  pages     = {209--223},
	  publisher = {Springer},
	  year      = {2020},
	  url       = {https://doi.org/10.1007/978-3-030-58583-9\_13},
	  doi       = {10.1007/978-3-030-58583-9\_13},
	  timestamp = {Tue, 01 Dec 2020 09:11:51 +0100},
	  biburl    = {https://dblp.org/rec/conf/eccv/BuiLZMDAP20.bib},
	  bibsource = {dblp computer science bibliography, https://dblp.org}
	}

## License  

As a free open-source implementation, ADR is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

ADR is licensed under the Apache License 2.0.