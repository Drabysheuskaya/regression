"""
Module providing functionality for polynomial regression, including data loading,
feature engineering, model training, and plotting results.
"""

import sys
from typing import Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def filter_data(data: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Remove rows from data array that contain NaN values."""
    return data[~np.isnan(data).any(axis=1)]


class PolynomialRegression:
    """Class to handle polynomial regression analysis."""

    def __init__(self, file_path: str, regression: str, degree: int) -> None:
        """Initialize the polynomial regression model with basic configuration."""
        self.input_file = file_path
        self.regression_type = regression.upper()
        self.poly_degree = degree
        self.equation: Optional[npt.NDArray[np.float_]] = None
        self.data: Optional[npt.NDArray[np.float_]] = None
        self.values: Optional[npt.NDArray[np.float_]] = None

    def load_data(self) -> npt.NDArray[np.float_]:
        """Load data from a CSV file specified by the input file path."""
        return np.loadtxt(self.input_file, delimiter=",", skiprows=1)

    def separate_features_and_target(self, data: npt.NDArray[np.float_]) -> None:
        """Separate input features and target values from the dataset."""
        self.values = data[:, -1]
        self.data = data[:, :1]

    def load_and_prepare_data(self) -> None:
        """Load data and prepare it by filtering and separating features/target."""
        data = self.load_data()
        data = filter_data(data)
        self.separate_features_and_target(data)

    def add_polynomial_features(self) -> None:
        """Generate polynomial features up to the specified degree for the model."""
        if self.data is not None:
            powers = np.arange(1, self.poly_degree + 1)
            self.data = np.power(self.data, powers)

    def fit(self) -> None:
        """Fit the regression model using the specified method (closed form or gradient)."""
        if self.regression_type == 'CLOSED':
            self.fit_closed_form()
        elif self.regression_type == 'GRADIENT':
            self.fit_gradient_based()
        else:
            raise ValueError(f"Unsupported regression type: {self.regression_type}")

    def fit_closed_form(self) -> None:
        """Fit the regression model using a closed-form solution."""
        if self.data is not None:
            x_biased = np.column_stack((self.data, np.ones((self.data.shape[0], 1))))
            x_biased_pseudo_inv = np.linalg.pinv(x_biased)
            self.equation = x_biased_pseudo_inv @ self.values

    def fit_gradient_based(self) -> None:
        """Fit the regression model using a gradient-based optimization method."""
        if self.data is not None:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.data)
            sgd = SGDRegressor(max_iter=1000, tol=1e-3)
            sgd.fit(data_scaled, self.values)
            coefficients = np.true_divide(sgd.coef_, scaler.scale_)
            self.equation = np.append(coefficients,
                                      sgd.intercept_ - np.dot(coefficients, scaler.mean_))

    def predict(self, datapoints: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Predict target values for the given datapoints using the trained model."""
        if self.equation is None or datapoints is None:
            return np.zeros((datapoints.shape[0],)) if datapoints is not None else None
        num_datapoints = datapoints.shape[0]
        datapoints_biased = np.column_stack((datapoints, np.ones(num_datapoints)))
        return datapoints_biased @ self.equation

    def plot_results(self) -> None:
        """Generate and save a plot of the actual data points and the model predictions."""
        if self.data is not None and self.values is not None:
            plt.plot(self.data[:, 0], self.values, color="red", marker=".",
                     linestyle="None", label="Actual Data")
            if self.equation is not None:
                predictions = self.predict(self.data)
                plt.plot(self.data[:, 0], predictions,
                         color="blue", linestyle="-", label="Predicted Line")
            plt.xlabel("Feature")
            plt.ylabel("Target")
            plt.legend()
            plt.savefig("regression_plot.png")

    def write_equation_to_file(self, filename: str = "regression_equation.txt") -> None:
        """Write the regression equation to a file in a human-readable form."""
        if self.equation is not None:
            equation_str = " + ".join(f"{coef} * X^{i + 1}"
                                      for i, coef in enumerate(self.equation[:-1]))
            equation_str += f" + {self.equation[-1]}"
            with open(filename, "wt", encoding='utf-8') as file:
                file.write(equation_str)
        else:
            print("Equation is not defined due to an error in fitting.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python regression.py <file_path> <regression> <degree>")
        sys.exit(1)

    input_file = sys.argv[1]
    regression_type = sys.argv[2]
    poly_degree = int(sys.argv[3])

    reg = PolynomialRegression(input_file, regression_type, poly_degree)
    reg.load_and_prepare_data()
    reg.add_polynomial_features()
    reg.fit()
    reg.write_equation_to_file()
    reg.plot_results()
