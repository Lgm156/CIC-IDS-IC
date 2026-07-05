#!/usr/bin/env bash

set -euo pipefail

models=(
    ft_transformer
    lstm
    logistic
    random_forest
    xgboost
)

for model in "${models[@]}"; do
    echo "Training $model..."

    python -m "scripts.train_${model}"
    python -m scripts.save_posteriors
    python -m "scripts.evaluate_${model}" >> finalResult.md

    echo "Finished $model"
    echo "------------------------"
done
