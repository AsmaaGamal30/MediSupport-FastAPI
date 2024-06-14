from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the Decision Tree model
model = joblib.load("./saved_models/decision_tree_model.pkl")


class HeartData(BaseModel):
    BMI: float
    PhysicalHealth: float
    MentalHealth: float
    SleepTime: float
    AgeCategory: int
    Race: int
    Diabetic: int
    GenHealth: int
    Sex: int
    Smoking: int
    AlcoholDrinking: int
    Stroke: int
    DiffWalking: int
    PhysicalActivity: int
    Asthma: int
    KidneyDisease: int
    SkinCancer: int


@app.post("/predict")
async def predict(data: HeartData):
    try:
        # Prepare input data for prediction
        input_data = np.array([[data.BMI, data.PhysicalHealth, data.MentalHealth, data.SleepTime, data.AgeCategory, data.Race,
                                data.Diabetic, data.GenHealth, data.Sex, data.Smoking,
                                data.AlcoholDrinking, data.Stroke, data.DiffWalking, data.PhysicalActivity,
                                data.Asthma, data.KidneyDisease, data.SkinCancer]])

        # Perform prediction using loaded model
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])  # Ensure output is an integer

        return {"prediction": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
