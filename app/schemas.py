from pydantic import BaseModel, Field
import warnings

warnings.filterwarnings("ignore")

class HeartDataInput(BaseModel):
    """Pydantic model defining raw input features for Heart Disease prediction."""

    age: float = Field(..., example=63.0, description="Age in years")
    sex: int = Field(..., example=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., example=1, description="Chest pain type (1, 2, 3, or 4)")
    trestbps: float = Field(..., example=145.0, description="Resting blood pressure in mm Hg")
    chol: float = Field(..., example=233.0, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., example=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",)
    restecg: int = Field(..., example=2, description="Resting electrocardiographic results")
    thalach: float = Field(..., example=150.0, description="Maximum heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise relative to rest",)
    slope: int = Field(..., example=3, description="Slope of peak exercise ST segment")
    ca: float = Field(..., example=0.0, description="Number of major vessels (0-3)")
    thal: float = Field(..., example=6.0, description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)",)


class PredictionResponse(BaseModel):
    """Pydantic model for API prediction response."""

    prediction: int = Field(..., description="Predicted class label (0 = No Risk, 1 = High Risk)")
    prediction_label: str = Field(..., description="Human-readable prediction result")
    confidence: float = Field(..., description="Confidence score / predicted probability for class 1")