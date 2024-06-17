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
#cfg = experiment.experiment.get_config(coadapt=False, cuda="1")

# AIP
#cfg = experiment.experiment.get_aip_lesion_config(cuda="3", coadapt=True)

# M1
cfg = experiment.experiment.get_m1_lesion_config(coadapt=False, cuda="3")

LOG_DIR = "/home/mbryan/coproc-poc/models"
my_coproc = cpn.CPN_EN_CoProc(cfg, log_dir=LOG_DIR)


my_experiment = experiment.experiment.stage(my_coproc, cfg)

loss_history = my_experiment.run()

