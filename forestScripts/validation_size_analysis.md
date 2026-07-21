# Validation and Test Set Size Impact Analysis (Sizes 100 to 1200)

This report analyzes how the size of the validation/test set impacts the calibration results and the search accuracy of the Binary Search method vs. the Grid Search method.

| Size/Class | Total Val/Test | Binary Lambda | Binary Test Acc | Grid Lambda | Grid Test Acc | Diff (Binary - Grid) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 700 | 2.18 | 0.737143 | 0.95 | 0.841429 | -0.104286 | Grid Better |
| 200 | 1400 | 1.19 | 0.812143 | 0.78 | 0.830714 | -0.018571 | Grid Better |
| 300 | 2100 | 1.88 | 0.759048 | 0.82 | 0.844286 | -0.085238 | Grid Better |
| 400 | 2800 | 1.23 | 0.823929 | 0.86 | 0.841071 | -0.017143 | Grid Better |
| 500 | 3500 | 1.08 | 0.846286 | 0.88 | 0.851143 | -0.004857 | Grid Better |
| 600 | 4200 | 0.54 | 0.818095 | 0.89 | 0.840000 | -0.021905 | Grid Better |
| 700 | 4900 | 1.12 | 0.834694 | 0.92 | 0.841837 | -0.007143 | Grid Better |
| 800 | 5600 | 0.91 | 0.836071 | 0.80 | 0.835357 | +0.000714 | Binary Better |
