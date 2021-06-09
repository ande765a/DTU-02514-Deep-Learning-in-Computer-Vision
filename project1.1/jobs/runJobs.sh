#!/bin/sh
bsub < ./01-baseline-sgd-no-aug.sh
bsub < ./02-baseline-adam-no-aug.sh
bsub < ./03-baseline-adam-aug.sh
bsub < ./03-resnet-adam-no-aug.sh
bsub < ./04-baseline-adam-aug.sh
bsub < ./05-resnet-adam-aug.sh