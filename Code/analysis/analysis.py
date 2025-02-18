import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from config import path_to_repository


class bender_class:
    
    '''
    class to manage data loading from CSVs, model training and testing, and data plotting
    '''

    def __init__(self, path=None):
        '''
        Initialize values for data, accuracy, model, and polynomial features
        '''
        self.data = None  # dataframe containing data from all csv files analyzed -> m rows by 4 columns
        self.acc = None  # accuracy from quadratic curve fitting class method:  quadriatic_fit(self)
        self.model = None  # To store the trained model
        self.poly_features = None  # To store polynomial features
        self.model_types = None # model-ftting type
        self.all_accuracies = []  # Initialize as an empty list for collecting accuracies
        self.accuracy_angle = np.arange(1, 16)  # Angle thresholds for accuracy calculations

        if path is None:
            self.repo_path = path_to_repository
        else: 
            self.repo_path = path

    def __str__(self):
        """
        human-readable, or informal, string representation of object
        """
        return (f"Bender Class: \n"
                f"  Number of data points: {self.data.shape[0] if self.data is not None else 0}\n"
                f"  Number of features: {self.data.shape[1] if self.data is not None else 0}\n"
                f"  Current Accuracy: {self.acc:.2f}% if self.acc is not None else 'N/A'\n")

    def __repr__(self):
        """
       more information-rich, or official, string representation of an object
       """

        return (f"Bender_class(data={self.data.head() if self.data is not None else 'None'}, "
                f"acc={self.acc}, "
                f"model={self.model.__class__.__name__ if self.model else 'None'}, "
                f"poly_features={self.poly_features.__class__.__name__ if self.poly_features else 'None'})")

    def load_data(self, regex_path):
        '''
        method to load data from csv files that match the regular expression string "regex_path"
        '''

        # Check that regex_path is a string
        if not isinstance(regex_path, str):
            raise TypeError("Expected 'path' to be a string.")

        # Use glob to get all the files in the folder that match the regex pattern
        csv_files = glob.glob(regex_path)
        print(csv_files)

        # Check that csv_files is not empty
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the specified path: {regex_path}")

        # Load all the data
        dataframes = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue

            # Check if the DataFrame has exactly 4 columns
            if df.shape[1] != 4:
                raise ValueError(
                    f"Error: The file {f} does not contain exactly 4 columns. It has {df.shape[1]} columns.")

            # Remove rows where all columns equal "100" (usually first few rows)
            ix_ok = (df.iloc[:, 0] != 100) & (df.iloc[:, 1] != 100) & (df.iloc[:, 2] != 100) & (df.iloc[:, 3] != 100)
            df = df[ix_ok]

            # Convert rotary encoder to angle (degrees) -> ADC is Arduino Uno 10 bit (2**10 = 1024), rotary encoder has 320 degrees of rotation
            df['Rotary Encoder'] = df['Rotary Encoder'] * 320 / 1024

            # Shift rotary encoder angles to start tests at 0 degrees
            df['Rotary Encoder'] = df['Rotary Encoder'] - df['Rotary Encoder'].values[0]

            # make all rotary encoder angles > 0 so when plate is bent, it is at + 90 deg...if left alone angles go from 0 to -90 deg
            df['Rotary Encoder'] = df['Rotary Encoder'] * -1

            # Append the DataFrame to the list
            dataframes.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)

        # Add column names to make it so we dont need to remember the column numbers
        self.columns = self.data.columns

        # Have not yet normalized ADC data
        self.adc_normalized = False

    def normalize_adc_bw_01(self):
        """
        Normalizes ADC values to be between 0 and 1 while ensuring:
        - The first value starts at 0.
        - The data increases positively (mirrored if necessary).
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized:
            raise ValueError("ADC data already normalized.")

        # Get min and max ADC values
        max_val = self.data['ADC Value'].max()
        min_val = self.data['ADC Value'].min()

        # Normalize ADC values between 0 and 1
        self.data['ADC Value'] = (self.data['ADC Value'] - min_val) / (max_val - min_val)

        # Identify ADC at 0° and near 90° (or closest available)
        adc_at_0 = self.data.loc[self.data['Rotary Encoder'].idxmin(), 'ADC Value']
        adc_at_90 = self.data.iloc[(self.data['Rotary Encoder'] - 90).abs().idxmin()]['ADC Value']

        # Check if ADC values decrease with increasing angle
        if adc_at_90 < adc_at_0:
            self.data['ADC Value'] = 1 - self.data['ADC Value']  # Mirror the dataset

        # Mark as normalized
        self.adc_normalized = True
        self.normalize_type = 'MinMax --> 0-1'

        print('ADC normalized bw 0-1. ADC max: ', self.data['ADC Value'].max(), 'ADC min: ',
              self.data['ADC Value'].min())

    def normalize_adc_over_R0(self):
        """
        Normalize ADC values to (R - R₀) / R₀ where R₀ is the initial resistance at the first strain value.
        This ensures normalized resistance starts near zero at strain = 0.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized:
            raise ValueError("ADC data already normalized.")

        # Use the first ADC value as R₀
        R0 = self.data['ADC Value'].iloc[0]

        # Ensure R₀ is not zero to avoid division errors
        if R0 == 0:
            raise ValueError("Initial ADC value (R₀) is zero. Cannot normalize.")

        # Normalize the ADC values
        self.data['ADC Value'] = (self.data['ADC Value'] - R0) / R0

        # Mark as normalized
        self.adc_normalized = True
        self.normalize_type = '(R - R₀) / R₀'
        print(f"ADC normalized with initial value R₀: {R0}")

    def plot_data(self, scatter=False, title=''):
        """
        method to plot normalized ADC values vs Rotary Encoder angles (blue dots)
        option to do a scatter plot where color == sample index
        """
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        if self.adc_normalized == False: 
            raise ValueError("ADC data not normalized. Please normalize data first.")

        # Create scatter plot: 
        f, ax = plt.subplots()

        # Plotting Rotary Encoder data
        if scatter:
            # Set color equal to data.index 
            ax.scatter(self.data['Rotary Encoder'], self.data['ADC Value'], c=self.data.index, cmap='viridis', s=5,
                    label='Rotary Encoder')
        else:
            ax.plot(self.data['Rotary Encoder'], self.data['ADC Value'], 'b.', markersize=5,
                 label='Rotary Encoder')  # Blue dots for Rotary Encoder
        
        # Setting labels
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Normalized ADC \n %s'%self.normalize_type) # state normalization type
        ax.set_title(title)

        self.data_ax = ax; 
        self.data_fig = f

    def plot_mech_model_data(self, thick, l_ch, l_sam, area, res, scatter=False,
                             data_color='blue', model_color='green',
                             data_label='Experimental Data', model_label='Theoretical Model', ax=None):
        """
        Class method to plot normalized data (delta R / R₀) vs strain (ε) for both experimental data
        and a theoretical mechanics model. Supports custom legend names and plotting on the same axes.

        Args:
            thick (float): Thickness of the plate (inches).
            l_ch (float): Length of the channel (inches).
            l_sam (float): Length of the sample (inches).
            area (float): Cross-sectional area (m²).
            res (float): Initial resistance (ohms).
            scatter (bool): Whether to plot experimental data as a scatter plot.
            data_color (str): Color for the experimental data plot.
            model_color (str): Color for the theoretical model plot.
            data_label (str): Legend label for the experimental data.
            model_label (str): Legend label for the theoretical model.
            ax (matplotlib.axes.Axes): Existing axes object to plot on. If None, creates a new figure and axes.

        Raises:
            ValueError: If no data is loaded or the data is not normalized.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("No data loaded. Please load data using the load_data method.")

        # Ensure data is normalized
        if not self.adc_normalized:
            raise ValueError("Data not normalized. Please normalize the data using normalize_adc_over_R0().")

        # Compute strain (ε) for experimental data
        self.data['Strain (ε)'] = (thick * 0.0254) * (self.data['Rotary Encoder'] * np.pi / 180) / (l_sam * 0.0254)

        # Prepare theoretical model data
        theta = np.arange(0, np.pi / 2 + 0.1, 0.1)  # Include up to 90 degrees
        rho = 29.4 * 10 ** -8  # Electrical resistivity of galinstan
        eps_model = (thick * 0.0254) * theta / (l_sam * 0.0254)  # Strain (ε) for theoretical model
        dr_model = (rho * eps_model * (l_ch * 0.0254) * (8 - eps_model) /
                    ((area * 0.000645) * (2 - eps_model) ** 2))  # Resistance change
        drrt_model = dr_model / res  # Normalized resistance change for model

        # Create new plot or add to existing one
        if ax is None:
            fig, ax = plt.subplots()  # Create a new figure and axes if none provided

        # Plot experimental data
        if scatter:
            ax.scatter(self.data['Strain (ε)'], self.data['ADC Value'], c=self.data.index, cmap='viridis', s=5, label=data_label)
        else:
            ax.plot(self.data['Strain (ε)'], self.data['ADC Value'], '.', markersize=5, color=data_color,
                    label=data_label)

        # Plot theoretical model
        ax.plot(eps_model, drrt_model, '--', color=model_color, label=model_label)

        # Set labels and legend only if a new figure was created
        if ax.get_title() == '':
            ax.set_xlabel('Strain (ε)')
            ax.set_ylabel('Normalized ADC (ΔR / R₀)')
            ax.set_title('Experimental vs Theoretical Model')
        ax.legend()

        # Return the axes object to allow further plotting
        return ax

    def train_model_test_accuracy(self, perc_train=0.8, niter = 10):
        """
        class method that trains a linear model to predict rotary encoder angles from normalized ADC values
        """

        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        if self.adc_normalized == False:
            raise ValueError("ADC data not normalized. Please normalize data first.")

        self.accuracy_angle = np.arange(1, 16) # accuracy tested up to 15 deg
        self.accuracy = np.zeros((niter, len(self.accuracy_angle)))

        for i in range(niter): 
            # Cross-validation (train: 80%, test: 20%)
            # Shuffle -- don't want to be biased based on time 
            dataTrain, dataTest = train_test_split(self.data, test_size= 1.0 - perc_train, shuffle=True) 

            # X_train: normalized ADC values
            X_train = dataTrain['ADC Value'].values.reshape(-1, 1) # ADC values
            X_train = np.hstack((X_train, np.ones(X_train.shape)))  # Add a column of ones for the intercept term
            y_train = dataTrain['Rotary Encoder'].values  # Rotary Encoder angles

            # Fit a polynomial of degree X to the training data
            #poly_features = PolynomialFeatures(degree=1)  # degree of 1 corresponds to linear fit, 2 would be quadratic
            #X_train = -1 * dataTrain.iloc[:, 2].values.reshape(-1, 1)  # Reshapes the 1D array into a 2D array with one...
            # column and as many rows as needed for compatibility with sklearn functions
            #y_train = dataTrain.iloc[:, 3].values  # Converts Pandas Series to a NumPy array
            #X_train_poly = poly_features.fit_transform(X_train)

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            #self.poly_features = poly_features  # Store the polynomial features

            # Predicting using the test set
            X_test = dataTest['ADC Value'].values.reshape(-1, 1)  # Normalized ADC values
            X_test = np.hstack((X_test, np.ones(X_test.shape)))  # Add a column of ones for the intercept term
            Y_test = dataTest['Rotary Encoder'].values  # Rotary Encoder angles
            
            Y_pred = self.model.predict(X_test)

            for j, angle_accuracy in enumerate(self.accuracy_angle):
                self.accuracy[i, j] = self.accuracy_by_angle(Y_test, Y_pred, angle_accuracy)

            # Add this run's accuracy to the list
            self.all_accuracies.append(self.accuracy)

    def plot_trained_model_on_existing(self, ax=None, title=None):
        """
        Overlays the trained model predictions on an existing plot (from plot_data).

        - Requires that `plot_data()` has already been run and `ax` is provided.
        - Requires that `train_model_test_accuracy()` has been run first.

        Parameters:
        - ax (matplotlib.axes.Axes, optional): The existing plot to overlay the model curve.
        - title (str): Title of the plot (optional).

        If ax is not provided, it uses the stored `self.data_ax` from plot_data().
        """

        # Ensure data is available
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")

        # Ensure ADC values are normalized before plotting
        if not self.adc_normalized:
            raise ValueError(
                "ADC data is not normalized. Please normalize it using normalize_adc_over_R0() or normalize_adc_bw_01().")

        # Ensure model is trained
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please run train_model_test_accuracy first.")

        # Use existing plot if no ax is provided
        if ax is None:
            if not hasattr(self, 'data_ax') or self.data_ax is None:
                raise ValueError("No existing plot found. Please run plot_data() first or provide an ax object.")
            ax = self.data_ax  # Use the stored axis from plot_data()

        # Generate model predictions across a range of ADC values
        adc_values = np.linspace(self.data['ADC Value'].min(), self.data['ADC Value'].max(), 100).reshape(-1, 1)
        adc_values = np.hstack((adc_values, np.ones(adc_values.shape)))  # Add intercept term

        # Predict angles using the trained model
        predicted_angles = self.model.predict(adc_values)

        # Plot the model's predicted curve on the existing ax
        ax.plot(predicted_angles, adc_values[:, 0], color='red', linestyle='--', linewidth=2, label="Trained Linear Model")

        # Update title if necessary
        ax.set_title(title)

        # Add legend if not already present
        ax.legend()

        # Show the updated plot
        plt.show()

    from sklearn.metrics import mean_squared_error
    import numpy as np

    def cross_validation_angular_error(self):
        """
        Performs 10-fold cross-validation by splitting the data into 10 parts.
        - Each fold uses 9/10 of the data for training and 1/10 for testing.
        - Predicts angles for the held-out data points.
        - Computes the mean and standard deviation of the angular error across all folds.

        Returns:
            mean_error (float): Mean of the absolute angular errors.
            std_error (float): Standard deviation of the angular errors.
            predictions_df (pd.DataFrame): DataFrame with actual and predicted angles.
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if not self.adc_normalized:
            raise ValueError("ADC data is not normalized. Please normalize it first.")

        N = len(self.data)
        num_splits = 10
        indices = np.arange(N)
        np.random.shuffle(indices)  # Shuffle the indices for randomness
        split_size = N // num_splits  # Number of data points per split

        all_predictions = np.zeros(N)  # Store predictions for each data point
        all_errors = np.zeros(N)  # Store errors

        for i in range(num_splits):
            # Define held-out and training indices
            test_indices = indices[i * split_size:(i + 1) * split_size]
            train_indices = np.setdiff1d(indices, test_indices)

            # Split data
            train_data = self.data.iloc[train_indices]
            test_data = self.data.iloc[test_indices]

            # Prepare training and testing inputs
            X_train = train_data['ADC Value'].values.reshape(-1, 1)
            X_train = np.hstack((X_train, np.ones(X_train.shape)))  # Add intercept term
            y_train = train_data['Rotary Encoder'].values

            X_test = test_data['ADC Value'].values.reshape(-1, 1)
            X_test = np.hstack((X_test, np.ones(X_test.shape)))  # Add intercept term

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict for the held-out data
            y_pred = model.predict(X_test)

            # Store predictions and errors
            all_predictions[test_indices] = y_pred
            all_errors[test_indices] = np.abs(y_pred - test_data['Rotary Encoder'].values)

        # Compute mean and std of the angular error
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)

        # Create DataFrame with actual and predicted values
        predictions_df = self.data.copy()
        predictions_df['Predicted Angle'] = all_predictions
        predictions_df['Absolute Error'] = all_errors


        return mean_error, std_error, predictions_df



    def cross_validation_external_test(self, external_datasets, n_splits=10):
        """
        Cross-validation method where:
        - One dataset is split into 10 parts (9N/10 for training, N/10 for testing).
        - The trained model is used to predict on the concatenated external datasets.
        - This process is repeated 10 times to compute mean and std errors.

        Parameters:
            external_datasets (list of bender_class instances): Other datasets to test on.
            n_splits (int): Number of cross-validation splits (default: 10).

        Returns:
            tuple: (mean_error, std_error, predictions_df)
                   - mean_error: Mean of absolute angular errors.
                   - std_error: Standard deviation of absolute angular errors.
                   - predictions_df: DataFrame with true angles and predicted angles.
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized is False:
            raise ValueError("ADC data must be normalized before running cross-validation.")

        # Ensure external datasets are loaded and normalized
        external_data = []
        for dataset in external_datasets:
            if dataset.data is None:
                raise ValueError("One of the external datasets has no data loaded.")
            if dataset.adc_normalized is False:
                raise ValueError("All external datasets must be normalized before testing.")
            external_data.append(dataset.data)

        # Concatenate external datasets
        external_df = pd.concat(external_data, ignore_index=True)

        # Prepare K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True)
        errors = []

        predictions_list = []

        for train_idx, test_idx in kf.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]  # Held-out test set (N/10)

            # Train the model using the 9N/10 split
            X_train = train_data['ADC Value'].values.reshape(-1, 1)
            X_train = np.hstack((X_train, np.ones(X_train.shape)))  # Add intercept
            y_train = train_data['Rotary Encoder'].values

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            # Predict on the concatenated external dataset
            X_external = external_df['ADC Value'].values.reshape(-1, 1)
            X_external = np.hstack((X_external, np.ones(X_external.shape)))  # Add intercept
            y_true_external = external_df['Rotary Encoder'].values  # True external angles

            y_pred_external = self.model.predict(X_external)

            # Compute absolute error for this split
            abs_errors = np.abs(y_pred_external - y_true_external)
            errors.append(abs_errors)

            # Store predictions
            predictions_df = pd.DataFrame({
                'True Angle': y_true_external,
                'Predicted Angle': y_pred_external,
                'Absolute Error': abs_errors
            })
            predictions_list.append(predictions_df)

        # Combine all predictions into one DataFrame
        final_predictions_df = pd.concat(predictions_list, ignore_index=True)

        # Compute mean and std error
        mean_error = np.mean([np.mean(err) for err in errors])
        std_error = np.std([np.mean(err) for err in errors])

        return mean_error, std_error, final_predictions_df

    def plot_error_violin(self, error_dfs, labels=None):
        """
        Creates side-by-side violin plots for angular prediction errors from multiple datasets.

        Parameters:
            error_dfs (list of pd.DataFrame): List of DataFrames, each containing an 'Absolute Error' column.
            labels (list of str, optional): List of labels corresponding to each dataset. Default is None.

        Raises:
            ValueError: If any DataFrame does not contain the expected column.
        """

        # Check if input is a list of DataFrames
        if not isinstance(error_dfs, list) or not all(isinstance(df, pd.DataFrame) for df in error_dfs):
            raise ValueError("error_dfs must be a list of pandas DataFrames.")

        # Ensure all DataFrames contain the 'Absolute Error' column
        for df in error_dfs:
            if "Absolute Error" not in df.columns:
                raise ValueError("Each DataFrame must contain an 'Absolute Error' column.")

        # Create a combined DataFrame for plotting
        plot_data = []
        dataset_labels = []

        for i, df in enumerate(error_dfs):
            plot_data.extend(df["Absolute Error"].tolist())  # Store error values
            dataset_label = labels[i] if labels and i < len(labels) else f"Dataset {i + 1}"
            dataset_labels.extend([dataset_label] * len(df))

        # Convert to DataFrame
        plot_df = pd.DataFrame({"Absolute Error": plot_data, "Dataset": dataset_labels})

        # Create the violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Dataset", y="Absolute Error", data=plot_df, inner="point", palette="coolwarm")

        # Labels and title
        plt.xlabel("Datasets")
        plt.ylabel("Absolute Angular Error (degrees)")
        plt.title("Distribution of Angular Prediction Errors Across Datasets")
        plt.ylim(0, 15)
        # Show the plot
        plt.show()


    def accuracy_by_angle(self, y_true, y_pred, angle_accuracy):
        '''
        Method to calculate the accuracy of the model for specific thresholds of angle accuracy
        '''

        # Accuracy determined by finding the number of test data that predicts
        # actual angle correctly to within +/- deg_accuracy
        pos = np.sum(np.abs(y_true - y_pred) < angle_accuracy)
        total = len(y_true)

        return 100. * pos / total

    def plot_accuracy(self, accuracy = None, title=''):
       
        if not hasattr(self, 'accuracy'):
            assert(accuracy is not None)
        
        if accuracy is not None:
            pass
        else:
            accuracy = self.accuracy

        f, ax = plt.subplots()

        # Plotting accuracy
        ax.plot(self.accuracy_angle, np.mean(accuracy, axis=0), 'b-', label='Accuracy')
        ax.errorbar(self.accuracy_angle, np.mean(accuracy, axis=0), np.std(accuracy, axis=0), marker='|', color='k')
        ax.set_xlabel('Angle Accuracy (degrees)')
        ax.set_ylabel('Percent Accurate (held out data)')
        ax.set_title(title)
        ax.set_ylim([0, 100])

    def get_min_accuracy_100(self, accuracy_matrix=None):
        """
        Finds the smallest angle threshold where the mean accuracy reaches 100%.

        Parameters:
            accuracy_matrix (np.array, optional): If provided, uses this external accuracy data instead of self.accuracy.

        Returns:
            tuple: (min_angle_100, accuracy_value)
                   where min_angle_100 is the smallest angle where accuracy is 100%.
        """

        # Determine which accuracy matrix to use
        if accuracy_matrix is None:
            if not hasattr(self, 'accuracy') or self.accuracy is None:
                raise ValueError("Accuracy has not been computed. Please run train_model_test_accuracy first.")
            accuracy_matrix = self.accuracy  # Default to self.accuracy

        # Compute mean accuracy across runs
        mean_accuracy = np.mean(accuracy_matrix, axis=0)

        # Find the first occurrence where accuracy reaches 100%
        indices_100 = np.where(mean_accuracy == 100)[0]

        if len(indices_100) == 0:
            print("No angle threshold reaches 100% accuracy.")
            return None, None

        # Get the smallest angle threshold where accuracy is 100%
        min_angle_100 = self.accuracy_angle[indices_100[0]]
        accuracy_value = mean_accuracy[indices_100[0]]

        return min_angle_100, accuracy_value

    def plot_bar_chart(self, values, labels, title="Min Angle for 100% Accuracy", ylabel="Min Angle (deg)", color="b",
                       ylim=(0, 15)):
        """
        Creates a bar chart where you can add as many bars as needed.

        Parameters:
            values (list): Data to plot (e.g., min angle for 100% accuracy).
            labels (list): Labels corresponding to each dataset.
            title (str): Title of the plot (default: "Min Angle for 100% Accuracy").
            ylabel (str): Label for the y-axis (default: "Min Angle (deg)").
            color (str): Bar color (default: 'b' for blue).
            ylim (tuple): Y-axis limits (default: (0, 15)).
        """

        if not (len(values) == len(labels)):
            raise ValueError("values and labels must have the same length.")

        # Set figure size
        plt.figure(figsize=(10, 5))

        # Define bar positions
        x_pos = np.arange(len(labels))

        # Create bar plot
        plt.bar(x_pos, values, color=color, alpha=0.7)

        # Set labels and title
        plt.xticks(x_pos, labels, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(ylim)  # Set y-axis range

        # Show plot
        plt.show()


    def plot_double_bar_chart(self, list1, list2, labels,
                              std1=None, std2=None,
                              title1="RMSE per Sample", title2="Min Angle for 100% Accuracy",
                              ylabel1="RMSE (deg)", ylabel2="Min Angle (deg)",
                              color1='b', color2='r', ylim1=(0, 15), ylim2=(0, 15)):
        """
        Creates two stacked bar plots for visualizing RMSE and Min Angle where Mean Accuracy = 100,
        with optional error bars.

        Parameters:
        - list1 (list): Data for the first bar plot (e.g., RMSE values).
        - list2 (list): Data for the second bar plot (e.g., min angle where accuracy is 100%).
        - labels (list): Labels for the bars (should be the same length as list1 and list2).
        - std1 (list, optional): Standard deviations for the first bar plot (RMSE) (default: None).
        - std2 (list, optional): Standard deviations for the second bar plot (Min Angle) (default: None).
        - title1 (str): Title for the first bar plot.
        - title2 (str): Title for the second bar plot.
        - ylabel1 (str): Y-axis label for the first plot.
        - ylabel2 (str): Y-axis label for the second plot.
        - color1 (str): Color for the first bar plot (default: 'b' for blue).
        - color2 (str): Color for the second bar plot (default: 'r' for red).
        - ylim1 (tuple): Y-axis limits for the first plot (default: (0, 15)).
        - ylim2 (tuple): Y-axis limits for the second plot (default: (0, 15)).
        """

        # Ensure input lists have the same length
        if not (len(list1) == len(list2) == len(labels)):
            raise ValueError("list1, list2, and labels must have the same length.")

        # Convert std to None if not provided to avoid errors in error bars
        if std1 is None:
            std1 = [0] * len(list1)  # No error bars if std1 is not provided
        if std2 is None:
            std2 = [0] * len(list2)  # No error bars if std2 is not provided

        # Ensure std lists are the same length as data lists
        if not (len(std1) == len(list1) and len(std2) == len(list2)):
            raise ValueError("Standard deviation lists must have the same length as their corresponding data lists.")

        # Set figure size
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Define bar positions
        x_pos = np.arange(len(labels))

        # Create first bar plot (RMSE)
        axes[0].bar(x_pos, list1, color=color1, alpha=0.7, yerr=std1, capsize=5, error_kw={'elinewidth': 1.5})
        axes[0].set_ylabel(ylabel1)
        axes[0].set_title(title1)
        axes[0].set_ylim(ylim1)  # Customizable y-axis limit
        axes[0].grid(axis='y', linestyle='--', alpha=0.6)  # Adds grid lines for readability

        # Create second bar plot (Min Angle where Accuracy = 100)
        axes[1].bar(x_pos, list2, color=color2, alpha=0.7, yerr=std2, capsize=5, error_kw={'elinewidth': 1.5})
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel(ylabel2)
        axes[1].set_title(title2)
        axes[1].set_ylim(ylim2)  # Customizable y-axis limit
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)

        # Set x-axis labels
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def predict_new_data(self, new_data_df):
        """
        Uses result from trained model method from single dataset/test to make predictions on new data/test from another bender_class object
        and calculates the accuracy of those predictions 
        """
        # Ensure that the model has been trained
        if self.model is None: 
            raise Exception("Model has not been trained yet. Please run the train_test method first.")
        
        # Prepare new data for prediction
        X_new = new_data_df['ADC Value'].values.reshape(-1, 1) #-1 * new_data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the same structure
        X_new = np.hstack((X_new, np.ones((X_new.shape[0], 1))))
        y_new = new_data_df['Rotary Encoder'].values  # Assuming the same structure

        # Predict using the trained model
        y_pred_new = self.model.predict(X_new)

        # Calculate accuracy based on the setpoint
        # Assuming the actual angles are in the fourth column of the new data
        accuracy = np.zeros((len(self.accuracy_angle)))
        for j, angle_accuracy in enumerate(self.accuracy_angle):
            accuracy[j] = self.accuracy_by_angle(y_new, y_pred_new, angle_accuracy)

        return accuracy

    def plot_combined_accuracy(self, title='Combined Accuracy vs Angle'):
        """
        Combine all accuracy plots into one showing average accuracy and standard deviation as a plot with error bars
        """
        if not self.all_accuracies or len(self.all_accuracies) == 0:
            raise ValueError("No accuracy data available. Train and test the model first.")

        # Concatenate all accuracy arrays
        all_accuracies_combined = np.vstack(self.all_accuracies)  # Shape: (runs * niter, len(accuracy_angle))
        print(len(all_accuracies_combined))

        # Calculate mean and standard deviation
        mean_accuracy = np.mean(all_accuracies_combined, axis=0)
        std_dev = np.std(all_accuracies_combined, axis=0)

        # Plot
        f, ax = plt.subplots()
        accuracy_angle = self.accuracy_angle  # Use angle thresholds from the class
        ax.plot(self.accuracy_angle, mean_accuracy, 'k-', label='Average Accuracy')  # Mean accuracy as a blue line
        ax.errorbar(self.accuracy_angle, mean_accuracy, std_dev,  marker='|', color='k')  # Mean + STD
        ax.set_xlabel('Angle Accuracy (degrees)')
        ax.set_ylabel('Percent Accurate')
        ax.set_ylim([0, 100])
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def dynamic_data(self, period):
        """
        Code used to extract timestamp info from dynamic autobending test.  After plate would bend 90 deg,
        'period' seconds wait before collecting 15 data points. Same when going back to 0 degrees.
        Test was conducted over 100 cycles.

        """
        # Add timestamp column
        num_rows = self.data.shape[0]  # Get the number of rows
        timestamps = pd.Series(range(num_rows)) * period  # 0.3 seconds (300 ms)
        self.data['Timestamp'] = timestamps

        # Ensure no mismatch in lengths
        self.data['ADC Value'] = self.data['ADC Value'].dropna()  # Remove NaN values in the y-data
        self.data['Timestamp'] = self.data['Timestamp'][:len(self.data['ADC Value'])]  # Match the lengths of x and y

    def plot_dynamic(self, time):

        """
        Code used to plot data from dynamic autobending test.
        'time' is the time domain to plot in 2nd subplot.

        """

        # Create subplots (1 row, 2 columns)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot the full range on the first subplot
        ax[0].plot(self.data['Timestamp'], self.data['ADC Value'], 'b')
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('$\Delta R/R_o$')
        ax[0].set_title('Full Plot')

        # Zoomed-in plot for time range between 0 and 4 seconds
        ax[1].plot(self.data['Timestamp'], self.data['ADC Value'], 'b')
        ax[1].set_xlim(0, time)  # Set x-axis to zoom between 0 and 4 seconds
        ax[1].set_xlabel('Time (sec)')
        ax[1].set_ylabel('$\Delta R/R_o$')
        ax[1].set_title('Zoomed-in Plot')

        # Display the subplots
        plt.tight_layout()  # Adjust the spacing
        plt.show()


class original_bender_class:  
    
    def __init__(self, data):
        
        # Ensure data is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected 'data' to be a pandas DataFrame.")
        
        self.data = data #dataframe containing data from all csv files analyzed -> m rows by 4 columns
        self.acc = None  #accuracy from quadratic curve fitting class method:  quadriatic_fit(self)
        self.model = None  # To store the trained model
        self.poly_features = None  # To store polynomial features
        
    def __str__(self):
        """
        human-readable, or informal, string representation of object
        """
        return (f"Bender Class: \n"
            f"  Number of data points: {self.data.shape[0] if self.data is not None else 0}\n"
            f"  Number of features: {self.data.shape[1] if self.data is not None else 0}\n"
            f"  Current Accuracy: {self.acc:.2f}% if self.acc is not None else 'N/A'\n")
    
    def __repr__(self):
        """
        more information-rich, or official, string representation of an object
        """
        return (f"Bender_class(data={self.data.head() if self.data is not None else 'None'}, "
            f"acc={self.acc}, "
            f"model={self.model.__class__.__name__ if self.model else 'None'}, "
            f"poly_features={self.poly_features.__class__.__name__ if self.poly_features else 'None'})")
    
    def read_data(self, path):
        """
        NOT USED IN LATEST ANALYSIS:  class method to extract all csv files in path and concatenate data in pandas dataframe
        class method also normalizes ADC around 0 and converts rotary encoder angle to deg, then 
        shifts angles to start at 0 deg
        """
        
        # use glob to get all the csv files
        # in the folder
        csv_files = glob.glob(path)
        
        if not isinstance(path, str):
            raise TypeError("Expected 'path' to be a string.")
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the specified path: {path}")
        
        # Initialize an empty list to hold the DataFrames
        dataframes = []

        # loop over the list of csv files
        for f in csv_files:
                  
            try:
                # Read the csv file
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
            
            # Check if the DataFrame has exactly 4 columns
            if df.shape[1] != 4:
                raise ValueError(f"Error: The file {f} does not contain exactly 4 columns. It has {df.shape[1]} columns.")
            
            # Remove rows where the first column equals 100
            df = df[df.iloc[:, 0] != 100]
            df = df[df.iloc[:, 1] < 95]
            
            # center ADC values around 0 (normalize ADC values)
            #df.iloc[:, 2] = (df.iloc[:, 2] - df.iloc[1, 2]) / df.iloc[1, 2]

            
            #Changes made 10/14/2024:  Normalization of data from 0 -> 1:  
            #https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            
            y = df.iloc[:, 2].values #returns a numpy array
            y = y.reshape(-1, 1)
            # Initialize the MinMaxScaler
            min_max_scaler = preprocessing.MinMaxScaler()
            # Fit and transform the data
            y_scaled = min_max_scaler.fit_transform(y)
            # Update the DataFrame with scaled values
            df.iloc[:, 2] = y_scaled.flatten()  # Flatten back to 1D for assignment
                
            #convert rotary encoder to angle (deg) -> ADC is arduino Uno 10 bit
            df.iloc[:, 3] = df.iloc[:, 3] * 320 / 1024

            #shift rotary encoder angles to start tests at 0 deg
            df.iloc[:, 3] = df.iloc[:, 3] - df.iloc[1, 3]

            

            # Append the DataFrame to the list
            dataframes.append(df)
            
         # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)
        
        return self.data
    
    def read_data_norm(self, path):
        """
        Delta R / R_o  method: extracts all csv files in path and concatenate data in pandas dataframe
        class method also normalizes ADC around 0 and converts rotary encoder angle to deg, then 
        shifts angles to start at 0 deg
        """
    
        # use glob to get all the csv files in the folder
        csv_files = glob.glob(path)
    
        if not isinstance(path, str):
            raise TypeError("Expected 'path' to be a string.")
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the specified path: {path}")
    
        # Initialize an empty list to hold the DataFrames
        dataframes = []

        # loop over the list of csv files
        for f in csv_files:
              
            try:
                # Read the csv file
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
        
            # Check if the DataFrame has exactly 4 columns
            if df.shape[1] != 4:
                raise ValueError(f"Error: The file {f} does not contain exactly 4 columns. It has {df.shape[1]} columns.")
        
            # Remove rows where the first column equals 100
            df = df[df.iloc[:, 0] != 100]
       
                
        
            df = df[df.iloc[:, 1] > -93]
            # Ensure slopes are positive
            df.iloc[:, 2] = df.iloc[:, 2].abs()
        
            # Normalize column 2
            y = df.iloc[:, 2].values # Returns a numpy array
            min_value = np.min(y)  # Find the smallest value in column 2
            y_change = (y - min_value) / min_value  # Calculate relative change from the smallest value
        
            # Update the DataFrame with the calculated change
            df.iloc[:, 2] = y_change  # Replace original values with relative change
        
            # Convert rotary encoder to angle (deg)
            df.iloc[:, 3] = df.iloc[:, 3] * 320 / 1024

            # Shift rotary encoder angles to start tests at 0 deg
            df.iloc[:, 3] = df.iloc[:, 3] - df.iloc[1, 3]

            # Append the DataFrame to the list
            dataframes.append(df)
    
        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)
    
        # Plot the change in column 2 over the smallest value
        #plt.plot(self.data.iloc[:, 3], self.data.iloc[:, 2], marker='o', label='Change in Column 2')
        #plt.xlabel('Rotary Encoder Angle (deg)')
        #plt.ylabel('$\Delta R/R_o$')
        #plt.legend()
        #plt.grid()
        #plt.show()
    
        return self.data
    
    def read_data_2(self, path):
        """
        MinMax normalization method:  Method extracts all csv files in path and concatenate data in pandas dataframe.
        Y-axis data range from 0 to 1, converts rotary encoder angle to degrees.  Data also drops NaN rows and has the potential to get rid of 
        very extreme data points (looks like a handful in each dataset)!
        """

        # Use glob to get all the csv files in the folder
        csv_files = glob.glob(path)
        print(f"Found {len(csv_files)} CSV files.")  # Debug statement
    
        # Initialize an empty list to hold the DataFrames
        dataframes = []

        # Loop over the list of csv files
        for f in csv_files:
            df = pd.read_csv(f)
        
            # Remove rows where the first column equals 100
            df = df[df.iloc[:, 0] != 100]
            
                    
            # Remove rows with any NaN values
            df = df.dropna()
        
            # center ADC values around 0 (normalize ADC values)
            #df.iloc[:, 2] = (df.iloc[:, 2] - df.iloc[15, 2]) / df.iloc[1, 2]
        
             # Ensure slopes are positive
            df.iloc[:, 2] = df.iloc[:, 2].abs()
            
            #Changes made 10/14/2024:  Normalization of data from 0 -> 1:  
            #https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            
            y = df.iloc[:, 2].values #returns a numpy array
            y = y.reshape(-1, 1)
            # Initialize the MinMaxScaler
            min_max_scaler = preprocessing.MinMaxScaler()
            # Fit and transform the data
            y_scaled = min_max_scaler.fit_transform(y)
            # Update the DataFrame with scaled values
            df.iloc[:, 2] = y_scaled.flatten()  # Flatten back to 1D for assignment
                
        
            # Convert rotary encoder to angle (degrees) -> ADC is Arduino Uno 10 bit
            df.iloc[:, 3] = df.iloc[:, 3] * 320 / 1024
        
            # Shift rotary encoder angles to start tests at 0 degrees
            initial_angle = df.iloc[20, 3]  # Save initial angle for later adjustment
            df.iloc[:, 3] = df.iloc[:, 3] - initial_angle 
            df = df[df.iloc[:, 3] > -93]
                 
            # Append the DataFrame to the list
            dataframes.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)

        return self.data
    
    
    
    def plot_data(self):
        """
        class method to plot normalized ADC values vs Rotary Encoder angles (blue dots) AND IMU angles (red dots)
        """
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        # Plotting Rotary Encoder data
        plt.plot(-1 * self.data.iloc[:, 3], self.data.iloc[:, 2], 'bo', label='Rotary Encoder')  # Blue circles for Rotary Encoder
        
        
        # Plotting IMU data
        #plt.plot(-1 * self.data.iloc[:, 1], self.data.iloc[:, 2], 'ro', label='IMU')  # Red circles for IMU
        #plt.yscale("log")
        #plt.xscale("log")
        
        # Setting labels
        plt.xlabel('Angle (deg)')
        plt.ylabel('MinMax(ADC)')

        # Adding legend
        plt.legend()
        plt.show()
        
        return
    
    
    def model_data(self, thick, l_ch, l_sam, area, res):
        """
        Class method to plot normalized data (delta R / Ro) vs bend angle as well as theoretical curve  based on mechanics model .
        """
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")

        # Plot Rotary Encoder data
        fig, ax1 = plt.subplots()

        ax1.plot(-1 * self.data.iloc[:, 3], self.data.iloc[:, 2], 'bo', label='Rotary Encoder Data')  # Blue circles for Rotary Encoder    

        theta = np.arange(0, np.pi/2 + 0.1, 0.1)  # Include 90 by adding increment
        rho = 29.4 * 10**-8
        eps = (thick * 0.0254) * theta / (l_sam * 0.0254)
        dr = (rho * eps * (l_ch * 0.0254) * (8 - eps) / ((area * 0.000645) * (2 - eps)**2))
        drrt = dr / res
        ax1.plot(theta * 180 / np.pi, drrt, 'g', label='Theoretical Model')  

        # Setting up bottom axis
        ax1.set_xlabel('Angle (deg)')
        ax1.set_ylabel('$\Delta R/R_o$', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc="upper left")

         # Define 5 evenly spaced tick positions for theta
        theta_ticks = np.linspace(0, np.pi/2, 5)  # 5 points from 0 to 90 degrees in radians
        eps_ticks = (thick * 0.0254 + 0.00635) * theta_ticks / (l_sam * 0.0254)  # Compute corresponding eps values

        # Create top x-axis for eps
        ax2 = ax1.twiny()  # Twin the x-axis to share the y-axis
        ax2.set_xlim(ax1.get_xlim())  # Synchronize with bottom x-axis

        ax1.set_xticks(theta_ticks * 180 / np.pi)  # Set bottom x-axis (theta) ticks in degrees
        ax2.set_xticks(theta_ticks * 180 / np.pi)  # Match top x-axis ticks to bottom x-axis
        ax2.set_xticklabels([f"{e:.2f}" for e in eps_ticks])  # Set top x-axis labels with eps values
        ax2.set_xlabel('$\epsilon$ (strain)')

        plt.tight_layout()
        plt.show()
        
    def train_test(self, deg_accuracy):
        """
        class method to determine how well a model that predicts angle based on normalized ADC value input up to +/- deg_accuracy 
        """
        
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
            
               
        # Cross-validation (train: 80%, test: 20%)
        dataTrain, dataTest = train_test_split(self.data, test_size=0.2, random_state=None)

        # Fit a polynomial of degree X to the training data
        poly_features = PolynomialFeatures(degree=1)  # degree of 1 corresponds to linear fit, 2 would be quadratic
        X_train = -1 * dataTrain.iloc[:, 2].values.reshape(-1, 1)  # Reshapes the 1D array into a 2D array with one... 
        # column and as many rows as needed for compatibility with sklearn functions
        y_train = dataTrain.iloc[:, 3].values  # Converts Pandas Series to a NumPy array
        X_train_poly = poly_features.fit_transform(X_train)

        self.model = LinearRegression()
        self.model.fit(X_train_poly, y_train)
        self.poly_features = poly_features  # Store the polynomial features

        # Predicting using the test set
        X_test = -1 * dataTest.iloc[:, 2].values.reshape(-1, 1)  
        X_test_poly = self.poly_features.transform(X_test)
        y_test = self.model.predict(X_test_poly)

        # Accuracy determined by finding the number of test data that predicts
        # actual angle correctly to within +/- deg_accuracy
        pos = np.sum(np.abs(y_test - dataTest.iloc[:, 3].values) < deg_accuracy)  
        total = len(dataTest)

        # Calculate accuracy
        self.acc = pos * 100 / total if total > 0 else 0  # Avoid division by zero
        
         # Optionally print the accuracy
        #print(f'Accuracy: {self.acc:.2f}%')
        
        return self.acc
    
    def train_test_log(self, deg_accuracy):
        """
        Class method to determine how well a model predicts angle based on 
        normalized ADC value input up to +/- deg_accuracy using log-log fitting.
        """
    
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        # Cross-validation (train: 80%, test: 20%)
        dataTrain, dataTest = train_test_split(self.data, test_size=0.2, random_state=None)

        # Prepare the training data using log transformation
        # Ensure we add a small constant to avoid log(0)
        X_train = -1 * dataTrain.iloc[:, 3].values + 1e-10  # Normalized ADC values
        y_train = dataTrain.iloc[:, 2].values +  1e-10        # Rotary Encoder angles
        
              
        # Fit a linear model to the log-log data
        self.model = LinearRegression()
        self.model.fit(X_train.reshape(-1, 1), y_train)  # Reshape for sklearn

        # Prepare the test data for predictions
        X_test = -1 * dataTest.iloc[:, 3].values + 1e-10  # Normalized ADC values
        y_test = dataTest.iloc[:, 2].values + 1e-10        # Rotary Encoder angles
        
       

        # Apply log transformation
        #X_test_log = np.log10(X_test)
        y_test_log_pred = self.model.predict(X_test.reshape(-1, 1))

        # Convert predictions back to original scale
        y_test_pred = 10 ** y_test_log_pred

        # Accuracy determined by finding the number of test data that predicts
        # actual angle correctly to within +/- deg_accuracy
        pos = np.sum(np.abs(y_test_pred - dataTest.iloc[:, 3].values) < deg_accuracy)  
        total = len(dataTest)

        # Calculate accuracy
        self.acc = pos * 100 / total if total > 0 else 0  # Avoid division by zero
    
        return self.acc
    
    def predict_new_data(self, new_data_obj, deg_accuracy):
        """
        Uses result from trained model method from single dataset/test to make predictions on new data/test from another bender_class object
        and calculates the accuracy of those predictions based on a specified degree of accuracy. 

        :param new_data_obj: An instance of bender_class containing new data for predictions.
        :param deg_accuracy: The degree of accuracy within which the predictions are considered correct.
        :return: A float representing the accuracy of the predictions.
        """
        # Ensure that the model has been trained
        if self.model is None or self.poly_features is None:
            raise Exception("Model has not been trained yet. Please run the train_test method first.")
        
        
        # Retrieve new data
        new_data = new_data_obj.data

        # Prepare new data for prediction
        X_new = -1 * new_data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the same structure
        X_new_poly = self.poly_features.transform(X_new)

        # Predict using the trained model
        y_pred_new = self.model.predict(X_new_poly)

        # Calculate accuracy based on the setpoint
        # Assuming the actual angles are in the fourth column of the new data
        actual_angles = new_data.iloc[:, 3].values
        pos = np.sum(np.abs(y_pred_new - actual_angles) < deg_accuracy)  # Count how many predictions are within the specified accuracy
        total = len(actual_angles)

        # Calculate accuracy
        accuracy = pos * 100 / total if total > 0 else 0  # Avoid division by zero

        # Optionally print the accuracy
        #print(f'Prediction Accuracy: {accuracy:.2f}%')

        # Create a DataFrame to hold the new data and the predictions
        results = pd.DataFrame(new_data)
        results['Predicted_Angle'] = y_pred_new  # Add predictions as a new column

        return accuracy, results

