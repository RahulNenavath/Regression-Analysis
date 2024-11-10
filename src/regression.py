import numpy as np
import pandas as pd
import scipy.stats as stats


class LinearRegressionNumpy:
    """
    A linear regression model implemented using NumPy.

    This class performs linear regression analysis with ordinary least squares (OLS)
    estimation. It provides functionality for fitting a linear model, calculating
    various regression statistics, and summarizing the results.

    Attributes:
        significance_level (float): Significance level for confidence intervals.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initializes the LinearRegressionNumpy class with a given significance level.

        Args:
            significance_level (float): The alpha value for calculating confidence intervals.
        """
        self.significance_level = significance_level
        self.coefficients = None
        self.covar_inv = None
        self.ssr = None
        self.sse = None
        self.sst = None
        self.s1_2 = None
        self.s2 = None
        self.s = None
        self.r2 = None
        self.r2_adj = None
        self.coeff_std_errors = None
        self.coeff_t_vals = None
        self.coeff_p_vals = None
        self.f_stat = None
        self.f_pval = None
        self.coeff_ci = None

    def _is_degenerate_matrix(self) -> bool:
        """
        Checks if the data matrix X is degenerate (has low rank).

        Returns:
            bool: True if the matrix is not degenerate; False otherwise.
        """
        rank = np.linalg.matrix_rank(self.X)
        self.full_rank = rank == min(self.rows, self.cols)
        self.low_rank = self.rows < self.cols
        self.degenerate = self.full_rank and not self.low_rank
        return not self.degenerate

    def _add_constant(self) -> None:
        """Adds a column of ones to the data matrix X to account for the intercept."""
        self.X = np.column_stack((np.ones(self.rows), self.X))

    def _standardize_features(self) -> None:
        """Standardizes the feature matrix X by subtracting mean and dividing by standard deviation."""
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def _closed_form_solution(self, X, y) -> np.ndarray:
        """
        Computes the closed-form solution of linear regression coefficients.

        Args:
            X (np.ndarray): The feature matrix with a constant column.
            y (np.ndarray): The response variable.

        Returns:
            np.ndarray: The regression coefficients.
        """
        self.covar_inv = np.linalg.inv(X.T @ X)
        return self.covar_inv @ X.T @ y

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the linear regression model to the provided data.

        Args:
            X (pd.DataFrame): The feature data with shape (n_samples, n_features).
            y (pd.Series): The target variable.

        Raises:
            ValueError: If the data matrix is degenerate and the model cannot be fit.
        """
        self.feature_names = X.columns
        self.X, self.y = X.to_numpy(), y.to_numpy()
        self.rows, self.cols = X.shape

        self.ddof_ssr = self.cols
        self.ddof_sse = self.rows - self.cols - 1
        self.ddof_sst = self.rows - 1

        self.X_bar = np.mean(self.X, axis=0)
        self.y_bar = np.mean(y)
        self.t_alpha_2 = stats.t.ppf(1 - self.significance_level / 2, self.ddof_sse)

        self._add_constant()

        if self._is_degenerate_matrix():
            self.coefficients = self._closed_form_solution(self.X, self.y)
        else:
            raise ValueError("Degenerate Data Matrix! Cannot fit Linear Model")

        self.y_hat = self.predict(self.X)

        self._compute_ssr()
        self._compute_sse()
        self._compute_sst()
        self._compute_s2()
        self._compute_r2()
        self._compute_coefficient_std_errors()
        self._compute_coefficient_t_statistic()
        self._compute_coefficient_t_stat_pvalue()
        self._compute_model_f_stat()
        self._compute_coefficient_confidence_interval()

    def _compute_ssr(self) -> None:
        """Calculates the regression sum of squares (SSR)."""
        self.ssr = np.sum((self.y_hat - self.y_bar) ** 2)
        self.s1_2 = self.ssr / self.ddof_ssr

    def _compute_sse(self) -> None:
        """Calculates the sum of squared errors (SSE)."""
        self.sse = np.sum((self.y_hat - self.y) ** 2)

    def _compute_sst(self) -> None:
        """Calculates the total sum of squares (SST)."""
        self.sst = self.ssr + self.sse

    def _compute_s2(self) -> None:
        """Calculates the variance estimate (s2) and residual standard error (s)."""
        self.s2 = self.sse / self.ddof_sse
        self.s = np.sqrt(self.s2)

    def _compute_r2(self) -> None:
        """Calculates the R-squared and adjusted R-squared metrics."""
        self.r2 = 1 - (self.sse / self.sst)
        self.r2_adj = 1 - ((self.sse / self.ddof_sse) / (self.sst / self.ddof_sst))

    def _compute_coefficient_std_errors(self) -> None:
        """Calculates the standard errors for each coefficient."""
        self.coeff_std_errors = [self.s * np.sqrt(self.covar_inv[idx][idx]) for idx in range(self.cols + 1)]

    def _compute_coefficient_t_statistic(self) -> None:
        """Calculates the t-statistics for each coefficient."""
        self.coeff_t_vals = [coeff / std_err for coeff, std_err in zip(self.coefficients, self.coeff_std_errors)]

    def _compute_coefficient_t_stat_pvalue(self) -> None:
        """Calculates the p-values for each coefficient's t-statistic."""
        self.coeff_p_vals = [2 * (1 - stats.t.cdf(abs(t_val), self.ddof_sse)) for t_val in self.coeff_t_vals]

    def _compute_model_f_stat(self) -> None:
        """Calculates the F-statistic and its corresponding p-value for the model."""
        self.f_stat = self.s1_2 / self.s2
        self.f_pval = 1 - stats.f.cdf(self.f_stat, self.ddof_ssr, self.ddof_sse)

    def _compute_coefficient_confidence_interval(self) -> None:
        """Calculates the confidence intervals for each coefficient."""
        self.coeff_ci = [
            (coeff - self.t_alpha_2 * std_err, coeff + self.t_alpha_2 * std_err)
            for coeff, std_err in zip(self.coefficients, self.coeff_std_errors)
        ]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the response variable for given feature matrix X.

        Args:
            X (np.ndarray): The feature matrix with shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values.
        """
        return X @ self.coefficients

    def summary(self):
        """
        Prints a summary of the regression model results similar to statsmodels' OLS summary.
        """
        print("OLS Regression Results")
        print("=" * 80)
        print(f"{'Dep. Variable:':<20} y                     {'R-squared:':<30} {self.r2:.3f}")
        print(f"{'Model:':<20} OLS                   {'Adj. R-squared:':<30} {self.r2_adj:.3f}")
        print(f"{'Method:':<20} Least Squares         {'F-statistic:':<30} {self.f_stat:.2f}")
        print(f"{'Date:':<20} {pd.Timestamp.now():%a, %d %b %Y} {'Prob(F-statistic):':<34} {self.f_pval:.2e}")
        print(f"{'Time:':<20} {pd.Timestamp.now():%H:%M:%S}")
        print("\nNo. Observations:              {:<12} Df Residuals: {:<10}".format(self.rows, self.ddof_sse))
        print(f"Df Model:                      {self.ddof_ssr}\n")
        print("Covariance Type:              nonrobust\n")

        # Coefficients Table Header
        print(f"{'':<10}{'coef':>10} {'std err':>10} {'t':>10} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}")
        print("-" * 80)

        # Coefficients Table Body
        for i, (coeff, std_err, t_val, p_val, ci) in enumerate(zip(
            self.coefficients, self.coeff_std_errors, self.coeff_t_vals, self.coeff_p_vals, self.coeff_ci)):
            feature = "const" if i == 0 else self.feature_names[i - 1]
            print(f"{feature:<10}{coeff:>10.4f} {std_err:>10.3f} {t_val:>10.3f} {p_val:>10.3f} {ci[0]:>10.3f} {ci[1]:>10.3f}")

        print("=" * 80)


if __name__ == '__main__':
    # Parameters for dataset
    n_samples = 100  # Number of samples
    n_features = 3   # Number of features

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate random feature data
    X = np.random.rand(n_samples, n_features) * 10  # Scale feature values

    # Define coefficients for the linear relationship
    true_coefficients = np.array([3.5, -2.7, 2.0])

    # Generate the target variable with some noise
    noise = np.random.randn(n_samples) * 2  # Add noise to the data
    y = X.dot(true_coefficients) + noise  # Linear relationship with noise

    # Convert to DataFrame for readability
    data = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    data['y'] = y

    features, target = data.drop('y', axis=1), data['y']

    num_rows = 10
    features = features[0:num_rows]
    target = target[0:num_rows]

    lr = LinearRegressionNumpy()
    lr.fit(features, target)
    lr.summary()