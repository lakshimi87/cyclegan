#!/bin/bash

datetime=$(date +"%Y%m%d_%H%M%S")
logfile="train.$datetime"

python train.py . 5011 >& "$logfile" &

