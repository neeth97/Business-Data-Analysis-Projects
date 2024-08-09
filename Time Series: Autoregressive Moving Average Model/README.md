## Objective
I use ACF and PACF graphs to analyze Catfish Sales data to fit an ARMA model.

## Analysis
**ACF:** Based on the ACF graph we see that the lag at 1 is pretty high but it shuts off at 2. I do see that they are high for 3 and 4 but I will start with MA(1) for now.

**PACF:** Based on PACF I see that the lag at 4 is the strongest and there are pretty strong lags at 12 showing a strong seasonality. So I start with AR(4)

**Model fitting:** I observe that the P values for all of them except for the constant are very low and hence significant. 

**Residuals:** I see that residuals are somewhat centered around 0 but are growing to absolute extreme values, that is something to watch out for.

**Actual vs Predictions:** I see that my predictions are mirroring the actual sales in the initial months but I do observe that they deviate from the actual sales later on. That is because I do not consider seasonality.
