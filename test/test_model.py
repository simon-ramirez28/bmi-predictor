import os
import pickle
import pytest
import pandas as pd
import numpy as np

MODEL_PATH = 'models/models_exported/bmi_model.pkl'

# --- Fixtures ---

@pytest.fixture
def model_path():
    """Returns the expected model path."""
    return MODEL_PATH


@pytest.fixture
def loaded_model():
    """Loads and returns the model if it exists."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


# --- Helper Functions ---

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height (cm) and weight (kg)."""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def create_input_data(height, weight, gender_encoded):
    """Create input DataFrame for the model."""
    bmi_value = calculate_bmi(height, weight)
    return pd.DataFrame({
        'Height': [height],
        'Weight': [weight],
        'BMI_Value': [bmi_value],
        'Gender_Encoded': [gender_encoded]
    })


# --- Test Cases ---

class TestModelLoading:
    """Tests for model loading functionality."""
    
    def test_model_file_exists(self, model_path):
        """Test that the model .pkl file exists."""
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    def test_model_loads_successfully(self, model_path):
        """Test that the model can be loaded from the .pkl file."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Model should not be None after loading"
    
    def test_model_has_predict_method(self, loaded_model):
        """Test that the loaded model has a predict method."""
        assert hasattr(loaded_model, 'predict'), "Model should have a predict method"
        assert callable(getattr(loaded_model, 'predict')), "Model predict should be callable"


class TestOutputRange:
    """Tests for BMI output range validation."""
    
    @pytest.mark.parametrize("height,weight,gender", [
        (170, 70, 0),   # Normal male
        (165, 60, 1),   # Normal female
        (180, 85, 0),   # Tall male
        (155, 55, 1),   # Short female
        (175, 75.5, 0), # With float weight
    ])
    def test_bmi_in_valid_range(self, loaded_model, height, weight, gender):
        """Test that calculated BMI is between 10 and 60 for normal data."""
        input_data = create_input_data(height, weight, gender)
        bmi_value = input_data['BMI_Value'].iloc[0]
        
        assert 10 <= bmi_value <= 60, f"BMI {bmi_value} is outside valid range [10, 60]"
    
    def test_prediction_returns_valid_index(self, loaded_model):
        """Test that model predictions are valid category indices (0-5)."""
        input_data = create_input_data(170, 70, 0)
        prediction = loaded_model.predict(input_data)[0]
        
        assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer"
        assert 0 <= prediction <= 5, f"Prediction index {prediction} is outside valid range [0, 5]"
    
    @pytest.mark.parametrize("height,weight,gender", [
        (140, 40, 0),   # Underweight case
        (200, 150, 1),  # Obesity case
        (160, 45, 0),   # Weak case
        (190, 120, 1),  # Extreme obesity case
    ])
    def test_various_bmi_categories_in_range(self, loaded_model, height, weight, gender):
        """Test that various BMI categories produce valid BMI values."""
        input_data = create_input_data(height, weight, gender)
        bmi_value = input_data['BMI_Value'].iloc[0]
        prediction = loaded_model.predict(input_data)[0]
        
        assert 10 <= bmi_value <= 60, f"BMI {bmi_value} is outside valid range"
        assert 0 <= prediction <= 5, f"Prediction {prediction} is outside valid range"


class TestTypeConsistency:
    """Tests for input type consistency."""
    
    def test_model_accepts_integers(self, loaded_model):
        """Test that the model accepts integer inputs."""
        input_data = create_input_data(170, 70, 0)
        
        # Verify inputs are integers
        assert isinstance(input_data['Height'].iloc[0], (int, np.integer))
        assert isinstance(input_data['Weight'].iloc[0], (int, np.integer))
        assert isinstance(input_data['Gender_Encoded'].iloc[0], (int, np.integer))
        
        # Model should predict without error
        prediction = loaded_model.predict(input_data)[0]
        assert 0 <= prediction <= 5
    
    def test_model_accepts_floats(self, loaded_model):
        """Test that the model accepts floating-point inputs."""
        input_data = create_input_data(170.5, 70.3, 0)
        
        # Verify inputs are floats
        assert isinstance(input_data['Height'].iloc[0], (float, np.floating))
        assert isinstance(input_data['Weight'].iloc[0], (float, np.floating))
        
        # Model should predict without error
        prediction = loaded_model.predict(input_data)[0]
        assert 0 <= prediction <= 5
    
    def test_model_accepts_mixed_types(self, loaded_model):
        """Test that the model accepts mixed int/float inputs."""
        bmi_value = calculate_bmi(170, 70.5)
        input_data = pd.DataFrame({
            'Height': [170],           # int
            'Weight': [70.5],          # float
            'BMI_Value': [bmi_value],  # float
            'Gender_Encoded': [1]      # int
        })
        
        # Model should predict without error
        prediction = loaded_model.predict(input_data)[0]
        assert 0 <= prediction <= 5
    
    @pytest.mark.parametrize("height,weight", [
        (170, 70),
        (170.0, 70.0),
        (170, 70.0),
        (170.0, 70),
    ])
    def test_various_type_combinations(self, loaded_model, height, weight):
        """Test various combinations of int and float inputs."""
        input_data = create_input_data(height, weight, 0)
        prediction = loaded_model.predict(input_data)[0]
        assert 0 <= prediction <= 5


class TestPredictionCategories:
    """Tests for prediction category validation."""
    
    def test_all_predictions_are_valid_categories(self, loaded_model):
        """Test that all possible predictions are valid category indices."""
        test_cases = [
            (140, 40, 0),   # Should be underweight (0 or 1)
            (170, 60, 1),   # Should be normal (2)
            (180, 90, 0),   # Should be overweight (3)
            (160, 100, 1),  # Should be obesity (4)
            (150, 120, 0),  # Should be extreme obesity (5)
        ]
        
        for height, weight, gender in test_cases:
            input_data = create_input_data(height, weight, gender)
            prediction = loaded_model.predict(input_data)[0]
            
            assert prediction in [0, 1, 2, 3, 4, 5], \
                f"Prediction {prediction} is not a valid category index"
    
    def test_prediction_consistency(self, loaded_model):
        """Test that the same input produces consistent predictions."""
        input_data = create_input_data(175, 75, 0)
        
        predictions = [loaded_model.predict(input_data)[0] for _ in range(5)]
        
        # All predictions should be the same
        assert all(p == predictions[0] for p in predictions), \
            "Model should produce consistent predictions for the same input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
