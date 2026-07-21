#!/bin/bash

python -m forestScripts.train_mlp

python -m forestScripts.save_posteriors

python -m forestScripts.evaluate_binary

python -m forestScripts.evaluate_grid
