#!/usr/bin/env python
# coding: utf-8

import logging
import os
import sys

l = logging.getLogger()
lh = logging.StreamHandler(sys.stdout)
l.addHandler(lh)
l.setLevel(logging.INFO)

import torch

import experiment
import cpn

# Connection
cfg = experiment.experiment.get_m1_lesion_config(coadapt=True, dont_train=True, cuda="1")


LOG_DIR = "/home/mbryan/coproc-poc/models/gaussian20.1.75_outputsIdxs0.50_gaussianExp16.sig2.175.decay0.3_enActTanh_cpnActTanh/d57bd818-f5f7-43d9-aef3-c075075f415d"
my_coproc = cpn.CPN_EN_CoProc(cfg, log_dir=LOG_DIR)


my_experiment = experiment.experiment.stage(my_coproc, cfg)

loss_history = my_experiment.run()

