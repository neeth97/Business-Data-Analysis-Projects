## Objective
We use ACF and PACF graph to analyse Catfish Sales data to fit a ARMA model.

## Analysis
**ACF:** Based on the ACF graph we see that the lag at 1 is pretty high but it shuts off at 2. We do see that they are high for 3 and 4 but we will start off with MA(1) for now.

**PACF:** Based on PACF we see that the lag at 4 is the strongest and there are pretty strong lags at 12 showing a  strong seasonality. So we start off with AR(4)

**Model fitting:** We observe that the P values for all of them expect for the constant are very low and hence significant. 

**Residuals:** We see that residuals are somewhat centred around 0 but are growing to absolute extreme values, that is something to watch out for.

**Actual vs Predictions:** We see that our predictions are mirroring the actual sales in the initial months but we do observe that they deviate from the actual sales later on. That is because we do not take seasonality into consideration.
