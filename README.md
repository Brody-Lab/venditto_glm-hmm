# GLM-HMM for binary observations
MATLAB functions for fitting a GLM-HMM on behavioral data  in tasks with a binary choice. Implementation based on Bishop's "Pattern Recognition and Machine Learning" and Escola et al. (2011), *Neural Computation*. To use, include both `fitGlmHmm.m` and `runBaumWelch.m` need to be in your MATLAB path.

## Model description
Coming soon

## Included functions
See function description and additional parameter descriptions by inputting `help fitGlmHmm` or `help runBaumWelch` into MATLAB command window.

### `fitGlmHmm`: Fitting the model
```
[model, ll] = fitGlmHmm(y,x,w0)
```
Required inputs:
- `y`: (1 x NTrials) binary observation data
- `x`: (NFeatures x NTrials) design matrix; behavioral features used to predict observation data
- `w0`: (NFeatures x NStates) initial latent state GLM weights. Desired number of latent states is taken implicitly from the second dimension

Outputs:
- `model`: struct containing fit parameters
  - `w`: (NFeatures x NStates) latent state GLM weights
  - `pi`: (NStates x 1) initial latent state probability
  - `A`: (NStates x Nstates) 
- `ll`: (1 x NIter) log-likelihood of model fit at each iteration



### `runBaumWelch`: Computing latent state probabilities or fit likelihood on some data set
```
[gammas,xis,ll] = runBaumWelch(y,x,model)
```
Required inputs:
- `y` and `x` as described above
- `model`: output of `runGlmHmm.m`

Outputs:
- `gammas`: probability of latent state given model parameters for each trial
- `xis`: joint posterior distribution (summed across trials). can be used to calculate the estimated transition matrix
- `ll`: log-likelihood of the model

## Example 
The MATLAB script `example_glmhmm_fit.m` details the process of fitting a GLM-HMM on an evidence accumulation ('accumulating towers') task data. This script:
1. Simulates a GLM-HMM model to generate behavioral data
2. Initializes necessary input and fits a GLM-HMM model to generated data (i.e. recovers GLM-HMM used to generate data)
3. Compares recovered fit to simulated model

## Credit
Sarah Jo Venditto
