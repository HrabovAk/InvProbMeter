import struct
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from inference import RealTimeInferenceNode

app = FastAPI()

# Ініціалізація моделі
try:
    inference_solver = RealTimeInferenceNode(
        model_path='best_solver_weights.pth', 
        scaler_path='scalers.pkl'
    )
    print("Модель успішно завантажено.")
except Exception as e:
    print(f"Помилка завантаження моделі: {e}")
    inference_solver = None

class ArduinoMessage(BaseModel):
    data: str # hex-рядок з Arduino

# 2. Ендпоінт для прийому даних
@app.post("/predict")
async def predict_endpoint(payload: ArduinoMessage):
    if inference_solver is None:
        return {"status": "SERVER ERROR", "msg": "MODEL NOT INITIALIZED"}

    try:
        binary_data = bytes.fromhex(payload.data)

        if len(binary_data) != 32:
            return {
                "status": "ERROR", 
                "msg": f"INVALID DATA LENGTH. Expected 32 bytes, got {len(binary_data)}"
            }
            
        # Розпаковка бінарних даних
        step_data = struct.unpack('<8f', binary_data)
        
        # Передаємо новий масив у модель
        result = inference_solver.process_new_step(list(step_data))
        
        # 3. Обробка результатів інференсу
        if result is None:
            return {
                "status": "WAITING",
                "msg": "BUFFERING OR NO GAS DETECTED"
            }
        else:
            params, _ = result
            
            return {
                "status": "SOLVED",
                "result": {
                    "D0": float(params[0]),
                    "Q0": float(params[1]),
                    "x0": float(params[2]),
                    "y0": float(params[3]),
                    "z0": float(params[4]),
                    "t0": float(params[5]),
                    "beta": float(params[6])
                }
            }
            
    except Exception as e:
        return {"status": "SERVER ERROR", "msg": str(e).upper()}

# 3. Ендпоінт для браузера
@app.get("/") # кореневий ендпоінт, який відобразить браузер при зверненні до сервера
async def home():
    return {"Qalification Work":"2026"}
