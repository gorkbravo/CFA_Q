import pandas as pd
from arch import arch_model

def garch_volatility_forecast(returns: pd.Series, p: int = 1, q: int = 1, vol: str = 'Garch') -> tuple[pd.Series, float]:
    """
    Fits a GARCH(p,q) model with a Student's t-distribution to the returns series 
    and returns the conditional volatility forecast and the degrees of freedom.
    """
    # Ensure returns are a Series with a DatetimeIndex
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Input Series must have a DatetimeIndex.")

    # Drop missing values and rescale
    returns = returns.dropna() * 100

    # Define the GARCH model with a Student's t-distribution
    model = arch_model(returns, vol=vol, p=p, q=q, dist='t')

    # Fit the model
    results = model.fit(disp='off')

    # Get the degrees of freedom
    dof = results.params['nu']

    # Return the conditional volatility (scaled back) and degrees of freedom
    return results.conditional_volatility / 100, dof
