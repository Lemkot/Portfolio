# module.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.dates as mdates


# Function to calculate all the metrics
def calculate_metrics(returns):
    """
    Calculate the business metrix from a series (a column of a DataFrame of logarithmic returns).

    Parameters:
    returns (DataFrame): a series (a column) of DataFrame containing logarithmic returns.

    Returns:
    Prints the buisness metrics.
    """
    # Calculate running maximum of cumulative returns
    cumulative_returns = np.exp(returns.cumsum())
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown series
    drawdown = (cumulative_returns - running_max)/running_max
    
    # Calculate max drawdown
    max_drawdown = drawdown.min()
    print("Max Drawdown:", max_drawdown)
    
    # Annualized return (portfolio)
    annualized_return = returns.mean() * 252
    print("Annualized Return:", annualized_return)
    
    # Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(252)
    print("Annualized Volatility:", annualized_volatility)
    
    # Annual Sharpe ratio (portfolio)
    sharpe_ratio = annualized_return / annualized_volatility
    print("Sharpe Ratio:", sharpe_ratio)
    
    # Calculate annualized Calmar ratio
    calmar_ratio = annualized_return / (abs(max_drawdown)) if abs(max_drawdown) > 0 else np.nan
    print("Calmar Ratio:", calmar_ratio)
    
    # Calculate Sortino ratio
    downside_returns = returns[returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)  # Annualize the downside deviation
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else np.nan
    print("Sortino Ratio:", sortino_ratio)
    
def plot_cumulative_returns_and_drawdowns(total_returns, x):
    """
    Plots cumulative returns

    Parameters:
    returns column (DataFrame): pands series = a coumn in a dataframe
    x: gradation of the y axis (float)

    Returns:
    Calculates cumulative returns and dawdowns and plots them in the same graph
    """
    # Calculate cumulative returns
    cumulative_returns = np.exp(total_returns.cumsum())
    cumulative_returns.index = pd.to_datetime(cumulative_returns.index)
    
    # Calculate rolling maximum and drawdowns
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    
    # Create figure and axis objects
    fig, ax = plt.subplots()
    
    # Plotting the results
    ax.plot(cumulative_returns - 1, label='Cumulative Returns')
    ax.fill_between(drawdowns.index, drawdowns, label='Drawdowns', color='red', alpha=0.3)
    
    # Setting x-axis major locator to each year and formatter
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Setting y-axis ticks every 20%
    ax.yaxis.set_major_locator(MultipleLocator(x))
    
    # Adding grid with vertical lines for each year
    ax.grid(True, which='major', linestyle='--', color='grey')
    
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    
    # Increasing font size of axis scale numbers
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Ensure date labels are visible
    ax.legend(fontsize=12)
    #ax.set_title('Cointegration-Based Strategy', fontsize=18)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Returns/Drawdown', fontsize=14)
    
    # Set the y-axis limits for consistency (optional)
    # ax.set_ylim(-0.8, 2.3)
    
    fig.tight_layout()
    plt.show()
    

def max_sharpe_ratio(returns):
    """
    Function to calculate portfolio weights that maximize the Sharpe Ratio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of each asset.
                            Each column represents an asset, and rows are returns for each time period.

    Returns:
    np.ndarray: The optimal weights for each asset that maximize the Sharpe Ratio.
    """
    # Calculate the mean returns and the covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Number of assets
    num_assets = len(mean_returns)

    # Define the objective function (negative Sharpe ratio, since we are minimizing)
    def objective(weights):
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio (assumed risk-free rate is 0)
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio  # Negative for minimization

    # Constraints: sum of weights = 1 (fully invested portfolio)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: each weight between 0 and 1
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess: equal allocation
    initial_guess = num_assets * [1. / num_assets]

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Return the optimal weights
    return result.x


def equal_weight(returns):
    """
    Function to assign equal weights to each asset in the portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of each asset.
                            Each column represents an asset, and rows are returns for each time period.

    Returns:
    np.ndarray: An array of equal weights for each asset.
    """
    # Number of assets
    num_assets = returns.shape[1]

    # Equal weights
    return np.ones(num_assets) / num_assets


def risk_parity(returns):
    """
    Function to calculate portfolio weights based on the Risk Parity strategy.
    This strategy assigns weights inversely proportional to the risk (volatility) of each asset.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of each asset.
                            Each column represents an asset, and rows are returns for each time period.

    Returns:
    np.ndarray: The portfolio weights based on Risk Parity strategy.
    """
    # Calculate the standard deviation (risk) of each asset
    risk = returns.std()

    # Inverse of the risk
    inverse_risk = 1 / risk

    # Normalize the weights so they sum to 1
    weights = inverse_risk / np.sum(inverse_risk)

    return weights


def minimum_volatility(returns):
    """
    Function to calculate portfolio weights based on the Minimum Volatility strategy.
    This strategy assigns weights to minimize the total volatility (standard deviation) of the portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of each asset.
                            Each column represents an asset, and rows are returns for each time period.

    Returns:
    np.ndarray: The portfolio weights based on the Minimum Volatility strategy.
    """
    # Calculate the covariance matrix of returns
    cov_matrix = returns.cov()

    # Number of assets
    num_assets = len(cov_matrix)

    # Define the objective function (minimize portfolio volatility)
    def objective(weights):
        # Portfolio variance
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_volatility

    # Constraints: sum of weights = 1 (fully invested portfolio)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: each weight between 0 and 1
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess: equal allocation
    initial_guess = num_assets * [1. / num_assets]

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Return the optimal weights
    return result.x

def rolling_window_portfolio(returns, opt_function, window_size=252, step_size=1):
    """
    This function calculates portfolio returns using a rolling window approach. 
    It applies an optimization function (e.g., risk parity) on a rolling window of returns 
    and generates portfolio weights.

    Arguments:
    returns -- DataFrame with asset returns
    window_size -- The size of the rolling window (number of days)
    opt_function -- The function to calculate asset weights from returns (e.g., risk parity)
    step_size -- The step size for moving the window (default is 1)

    Returns:
    portfolio_returns -- The portfolio returns after applying the optimization strategy
    weights -- DataFrame with portfolio weights for each window
    """
    
    # Initialize an empty DataFrame with the same shape and column names as the input data
    Y_hat_output = pd.DataFrame(index=returns.index, columns=returns.columns)

    n_rows = len(returns)
    start_idx = 0

    # Create the rolling window loop 
    for start_idx in range(0, n_rows - window_size, step_size):
        train_start = start_idx
        train_end = train_start + window_size
        test_start = train_end
        test_end = test_start + step_size
        X_train = returns.iloc[train_start:train_end]
        
        # Apply the optimization function on the training data
        weights = opt_function(X_train)
        
        # Use np.tile to repeat the weights for the test window
        Y_hat = np.tile(weights, (test_end - test_start, 1))
        
        # Create DataFrame for the predicted weights
        Y_hat_df = pd.DataFrame(Y_hat, index=returns.index[test_start:test_end], columns=returns.columns)
        
        # Assign the computed predictions to the appropriate locations in the output DataFrame
        Y_hat_output.loc[returns.index[test_start:test_end]] = Y_hat_df.values
        
        # Make sure the values are numeric (handle any non-numeric values)
        Y_hat_output = Y_hat_output.apply(pd.to_numeric, errors='coerce')
    
    # Copy the weights DataFrame to avoid altering the original
    weights = Y_hat_output.copy()

    # Calculate asset returns by multiplying the weights with returns
    asset_returns = weights * returns

    # Drop NaN values that result from non-overlapping windows
    asset_returns.dropna(inplace=True)

    # Calculate portfolio returns by summing the asset returns
    portfolio_returns = asset_returns.sum(axis=1)

    return portfolio_returns
