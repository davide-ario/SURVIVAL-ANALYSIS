
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DeepSurvivalNetwork:
    """
    A neural network model for survival analysis based on the DeepSurv approach (Katzman et al., 2018).
    This model uses a Cox Proportional Hazards loss function with a neural network
    to predict relative risk scores.
    """
    
    def __init__(self, input_dim, hidden_layers=[32, 16], dropout_rate=0.2):
        """
        Initialize the DeepSurvivalNetwork model.
        
        Args:
            input_dim: Dimensionality of the input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural network architecture."""
        input_layer = layers.Input(shape=(self.input_dim,))
        x = input_layer
        
        # Add hidden layers with dropout
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer (no activation - we want a linear risk score)
        risk_score = layers.Dense(1)(x)
        
        model = models.Model(inputs=input_layer, outputs=risk_score)
        return model
    
    def _negative_log_likelihood_loss(self, y_true, y_pred):
        """
        Custom loss function for Cox proportional hazards model.
        
        Args:
            y_true: Tensor of shape (batch_size, 2) containing [event_time, event_indicator]
            y_pred: Predicted risk scores
            
        Returns:
            Negative log likelihood loss
        """
        # Separate time and event indicator
        time = y_true[:, 0]
        event = y_true[:, 1]
        
        # Sort data by time
        indices = tf.argsort(time)
        event = tf.gather(event, indices)
        risk_scores = tf.gather(y_pred, indices)
        
        # Calculate loss
        # For each event, compute the log of the sum of exp(risk_scores) for all at-risk individuals
        # (those with equal or greater event times)
        time_sorted = tf.gather(time, indices)
        mask = tf.cast(time_sorted[:, None] <= time_sorted, dtype=tf.float32)
        
        # Risk set: exp(risk_scores) for subjects still at risk
        risk_set = tf.exp(risk_scores)
        
        # For each event, get the sum of exp(risk_scores) for all patients at risk
        cumulative_risk = tf.reduce_sum(risk_set * mask, axis=1)
        log_cumulative_risk = tf.math.log(cumulative_risk)
        
        # Only consider events (not censored)
        event_risk = risk_scores * event
        neg_likelihood = -tf.reduce_sum(event_risk - log_cumulative_risk * event)
        return neg_likelihood
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=64, patience=10):
        """
        Fit the DeepSurvival model.
        
        Args:
            X: Feature matrix
            y: Target matrix with columns [time, event]
            validation_split: Proportion of data to use for validation
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            History object
        """
        # Compile the model with custom loss
        self.model.compile(optimizer='adam', loss=self._negative_log_likelihood_loss)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Fit the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        return history
    
    def predict_risk(self, X):
        """
        Predict risk scores for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Risk scores
        """
        return self.model.predict(X)
    
    def plot_loss(self, history):
        """
        Plot training and validation loss.
        
        Args:
            history: History object from model.fit()
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients with some features being more important
    true_coefficients = np.random.randn(n_features) * np.array([3, 2, 1.5, 1, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05])
    
    # Baseline hazard and risk scores
    baseline_hazard = 0.01
    risk_scores = np.exp(np.dot(X, true_coefficients))
    
    # Generate survival times from exponential distribution
    event_times = np.random.exponential(scale=1.0 / (baseline_hazard * risk_scores))
    
    # Generate censoring times
    censor_times = np.random.exponential(scale=20, size=n_samples)
    
    # Observed time is the minimum of event time and censoring time
    observed_times = np.minimum(event_times, censor_times)
    
    # Event indicator (1 if event occurred, 0 if censored)
    event_indicators = (event_times <= censor_times).astype(int)
    
    # Combine time and event indicator
    y = np.column_stack((observed_times, event_indicators))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    survival_nn = DeepSurvivalNetwork(input_dim=n_features, hidden_layers=[32, 16, 8])
    history = survival_nn.fit(X_train_scaled, y_train, epochs=50, batch_size=64, patience=5)
    
    # Plot training history
    survival_nn.plot_loss(history)
    
    # Predict risk scores
    train_risk_scores = survival_nn.predict_risk(X_train_scaled)
    test_risk_scores = survival_nn.predict_risk(X_test_scaled)
    
    # Evaluate concordance index (a common metric for survival models)
    # Higher value means better discrimination
    from lifelines.utils import concordance_index
    
    c_index = concordance_index(
        y_test[:, 0],  # Event times
        -test_risk_scores.flatten(),  # Negative risk scores (higher means better survival)
        y_test[:, 1]   # Event indicators
    )
    
    print(f"Concordance Index on test data: {c_index:.4f}")
    
    # Plot risk scores by event status
    plt.figure(figsize=(10, 6))
    event_mask = y_test[:, 1] == 1
    plt.scatter(y_test[event_mask, 0], test_risk_scores[event_mask], 
                c='red', alpha=0.6, label='Event')
    plt.scatter(y_test[~event_mask, 0], test_risk_scores[~event_mask], 
                c='blue', alpha=0.6, label='Censored')
    plt.xlabel('Time')
    plt.ylabel('Predicted Risk Score')
    plt.title('Risk Scores vs. Observed Times')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Function to estimate baseline hazard
    def estimate_baseline_hazard(times, events, risk_scores):
        """
        Breslow estimator for the baseline hazard function.
        
        Args:
            times: Observed times
            events: Event indicators (1 for event, 0 for censored)
            risk_scores: Predicted risk scores
            
        Returns:
            time_points: Unique event times
            baseline_hazard: Estimated baseline hazard at each time point
            baseline_cumulative_hazard: Estimated baseline cumulative hazard at each time point
        """
        # Convert to numpy arrays if needed
        times = np.array(times)
        events = np.array(events)
        risk_scores = np.array(risk_scores).flatten()
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        events = events[sort_idx]
        risk_scores = risk_scores[sort_idx]
        
        # Get unique event times
        unique_times = np.unique(times[events == 1])
        
        # Calculate baseline hazard at each event time
        baseline_hazard = []
        risk_sum = []
        
        for t in unique_times:
            # Subjects at risk at time t (those with time >= t)
            at_risk = times >= t
            # Sum of risk scores for subjects at risk
            risk_sum_at_t = np.sum(np.exp(risk_scores[at_risk]))
            # Number of events at time t
            events_at_t = np.sum((times == t) & (events == 1))
            # Baseline hazard at time t
            if risk_sum_at_t > 0:
                baseline_hazard.append(events_at_t / risk_sum_at_t)
            else:
                baseline_hazard.append(0)
            risk_sum.append(risk_sum_at_t)
        
        # Calculate baseline cumulative hazard
        baseline_cumulative_hazard = np.cumsum(baseline_hazard)
        
        return unique_times, np.array(baseline_hazard), baseline_cumulative_hazard
    
    # Plot predicted hazard curves for selected individuals
    def plot_hazard_curves(times, events, risk_scores, n_individuals=3):
        """
        Plot hazard curves for selected individuals.
        
        Args:
            times: Observed times
            events: Event indicators
            risk_scores: Predicted risk scores
            n_individuals: Number of individuals to plot
        """
        # Estimate baseline hazard
        time_points, baseline_hazard, baseline_cumulative_hazard = estimate_baseline_hazard(times, events, risk_scores)
        
        # Select individuals with low, medium, and high risk
        sorted_indices = np.argsort(risk_scores.flatten())
        selected_indices = [
            sorted_indices[0],  # Lowest risk
            sorted_indices[len(sorted_indices) // 2],  # Medium risk
            sorted_indices[-1]  # Highest risk
        ]
        
        # Plot hazard curves
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Hazard function
        plt.subplot(2, 1, 1)
        for i, idx in enumerate(selected_indices):
            # Individual hazard = baseline hazard * exp(risk score)
            individual_hazard = baseline_hazard * np.exp(risk_scores[idx])
            risk_level = ["Low", "Medium", "High"][i]
            plt.plot(time_points, individual_hazard, label=f'{risk_level} Risk (Score: {risk_scores[idx][0]:.2f})')
        
        plt.title('Predicted Hazard Functions')
        plt.xlabel('Time')
        plt.ylabel('Hazard Rate')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Cumulative hazard function
        plt.subplot(2, 1, 2)
        for i, idx in enumerate(selected_indices):
            # Individual cumulative hazard = baseline cumulative hazard * exp(risk score)
            individual_cumulative_hazard = baseline_cumulative_hazard * np.exp(risk_scores[idx])
            risk_level = ["Low", "Medium", "High"][i]
            plt.plot(time_points, individual_cumulative_hazard, label=f'{risk_level} Risk (Score: {risk_scores[idx][0]:.2f})')
        
        plt.title('Predicted Cumulative Hazard Functions')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Hazard')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Subplot 3: Survival function
        plt.figure(figsize=(10, 6))
        for i, idx in enumerate(selected_indices):
            # Individual cumulative hazard = baseline cumulative hazard * exp(risk score)
            individual_cumulative_hazard = baseline_cumulative_hazard * np.exp(risk_scores[idx])
            # Survival function S(t) = exp(-H(t)) where H(t) is the cumulative hazard
            individual_survival = np.exp(-individual_cumulative_hazard)
            risk_level = ["Low", "Medium", "High"][i]
            plt.plot(time_points, individual_survival, label=f'{risk_level} Risk (Score: {risk_scores[idx][0]:.2f})')
        
        plt.title('Predicted Survival Functions')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Plot hazard curves for selected individuals in the test set
    plot_hazard_curves(y_test[:, 0], y_test[:, 1], test_risk_scores)
