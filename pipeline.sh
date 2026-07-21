#!/bin/bash

python -m forestScripts.train_mlp

python -m forestScripts.save_posteriors

python -m forestScripts.evaluate_binary >> binaryLambda.md

python -m forestScripts.evaluate_grid >> gridLambda.md

