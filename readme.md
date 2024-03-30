## CDS variant calculator and machine learning for multigene pathway optimization

  ### The multiple nonrepetitive coding sequence calculator (MNCC) was applied for generation of sets of coding DNA sequence (CDS) variants, which diverged from one another in terms of genetic composition without affecting protein expression in a specified host. P2n-All contained MNCC opensource was used for the constructing and the performing of CDS variants of three hosts (Y. lipolytica, S. cerevisiae, and E. coli), and pro.fasta was used for inputting amino acid sequences.

  ### Machine learning based on Gaussian Process Regressor (GPR) was used to rationally calculate and optimize the copy number combination for each gene of EPA and lycopene biosynthesis pathway. Opensource contained the Training and learning process of GPR. As shown in the Code Note, lowercase letter of example1 (a, b, c, d, e, f and g) represents genes of EPA and lowercase letter of example2 (a, b, c, d) lycopene biosynthesis pathway respectively.

#### Machine learning Project structure

  - example : Provided two benchmark data
  - - *.yaml  : Model configuration file
  - - - min_values : Minimum amount of enzyme
  - - - max_values ：Maximum amount of enzymes
  - - - repeat : Randomly generate number of combinations
  - - - top_k ： Target combination quantity
  - - - max_copy : Maximum portions of all enzymes in the combination
  - - *.csv : Data file
  - - - Notes, Please put the target content in the last column
  - output : Model output folder
  - utils : Model implementation folder
  - h_evaluate.py : Model evaluation code
  - h_predict.py : Prediction code
  - requirements.txt : Python environment must include list

## Machine learning How to use

```python
conda create --name gpr python=3.8
conda activate gpr
conda install --file requirements.txt

# Model selection
python h_evaluate.py 42 example/example1/data.csv


output is
seed: 42
category: 7
SVM MSE:         6.156
MLP MSE:         3.274
GPR MSE:         1.741

Note：Due to the random seed, there are slight differences in the values of MSE.


# Model predict
python h_predict.py 42 example/example2/config.yaml example/example2/data.csv

output is 
Seed: 42
Test Size: 0.2

GPR Predict(10/1000000):
         a  b   c   d  Predict_e
245889  12  7   9  12  89.837381
396000  11  7  10  12  89.536743
38364   12  7  10  11  89.314102
435009  10  7  11  12  89.236105
104186  11  7  11  11  89.013464
161020   9  7  12  12  88.935468
32935   12  7  11  10  88.790822
689519  10  7  12  11  88.712826
465779  11  7  12  10  88.490185
653361  12  8   8  12  88.363631
Average: 77.81462072068085
Std: 5.187133853308222
```
