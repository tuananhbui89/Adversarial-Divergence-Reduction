# Adversarial-Divergence-Reduction (ADR)

This pytorch version for the the paper ["Improving Adversarial Robustness by Enforcing Local and Global Compactness"](https://arxiv.org/abs/2007.05123) which has been accepted to ECCV-2020. [paper](https://arxiv.org/abs/2007.05123)[slide](https://www.dropbox.com/s/m7kdbte0rxh0qra/FIT_presentation_Sep_20.pdf?dl=0)

This also includes baselines methods: (1) PGD-AT (2) TRADES

## Requirements 
- Python 3.7
- Auto-Attack 0.1
- Foolbox 3.2.1
- Numba 0.52.0

## Robustness Evaluation 
We use several attackers to challenge the baselines and our method. 

(1) PGD Attack. We use the pytorch version of the Cifar10 Challenge with norm Linf, from the [implementation](https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py). Attack setting for the Cifar10 dataset: `epsilon`=8/255, `step_size`=2/255, `num_steps`=200, `random_init`=True 

(2) Auto-Attack. The official implementation in the [link](https://github.com/fra31/auto-attack). We test with the standard version with Linf

(3) Brendel & Bethge Attack. The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). It has been initialized with PGD attack (20 steps, `eta`=`epsilon`/2) to increase the success rate.  

We use the full test-set (10k images) for the attack (1) and 1000 first test images for the attacks (2-3).

## Training and Evaluation 

We provide the default setting for each corresponding dataset (MNIST, CIFAR10) which is used in our paper.

To reproduce the baselines, run the following script. 
```shell
python run_baseline.py
```

To reproduce our results, run the following script. 
```shell
python run_main.py
```

Please refer to the file `mysetting.py` for custom running. The pretrained model will be published soon. 

## References
- The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). 
- The Auto-attack is adapted from [the implementation](https://github.com/fra31/auto-attack) of the paper ["Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks", Francesco Croce, Matthias Hein, ICML 2020](https://arxiv.org/abs/2003.01690).

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