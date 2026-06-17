import os
import torch
import pickle
import numpy as np
from collections import deque
from model import Solver

class RealTimeInferenceNode:
    def __init__(self, model_path='best_solver_weights.pth', scaler_path='scalers.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Ініціалізація інференсу на: {self.device}")
        
        # Завантаження скейлерів
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Файл {scaler_path} не знайдено. Спочатку навчіть модель.")
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.feature_scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']
            
        # Ініціалізація моделі
        self.model = Solver(input_dim=8, output_dim=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Кільцевий буфер
        self.total_steps = 720                          # 720 кроків
        self.buffer = deque(maxlen=self.total_steps)    # Неперервне накопичування в фоні
        
        self.detection_threshold = 520.0                # Порогове значення в одиницях АЦП для старут інференсу
        self.dt = 10                                    # 10 секунд
        
        # Циклічний інференс кожні 60 секунд
        self.inference_delay_steps = 60 // self.dt       # 6 кроків для першого інференсу
        self.inference_interval_steps = 60 // self.dt    # 6 кроків між циклічними інференсами
        
        # Стани (чи є детект, кількість кроків після детекту та кількість кроків після останнього інференсу)
        self.is_detecting = False
        self.steps_since_detection = 0
        self.steps_since_last_inference = 0

    def process_new_step(self, step_data):
        self.buffer.append(step_data)
        
        # Перевірка перевищення порогу у 520.0 одиниць АЦП
        current_max_sensor = np.max(step_data[:4])
        gas_present = current_max_sensor > self.detection_threshold
        
        if gas_present:
            if not self.is_detecting:
                # Перший контакт з газовою хмарою
                self.is_detecting = True
                self.steps_since_detection = 1
                self.steps_since_last_inference = 0
                print(f"[ДЕТЕКТ] Помічено газ ({current_max_sensor:.1f}). Чекаємо 60 секунд для уточнення контексту...")
                return None
            else:
                self.steps_since_detection += 1
                self.steps_since_last_inference += 1
                
                # Умова 1: Пройшло 60 секунд після витоку (перший розрахунок)
                is_first_inference = (self.steps_since_detection == self.inference_delay_steps)
                
                # Умова 2: Цикл у 60 секунд, поки триває витік (подальші розрахунки)
                is_next_inference = (self.steps_since_detection > self.inference_delay_steps and 
                                     self.steps_since_last_inference == self.inference_interval_steps)
                
                if is_first_inference or is_next_inference:
                    self.steps_since_last_inference = 0
                    state_msg = "ПЕРШИЙ" if is_first_inference else "ЦИКЛІЧНИЙ"
                    print(f"[{state_msg} ІНФЕРЕНС] Gas in the air for {self.steps_since_detection * self.dt} sec. Updating parameters...")
                    return self._run_inference()
        else:
            if self.is_detecting:
                print("[ОЧИСТКА] Концентрація впала нижче порогу. Реалтайм-інференс призупинено.")
                self.is_detecting = False
                self.steps_since_detection = 0
                self.steps_since_last_inference = 0
                
        return None

    def _run_inference(self):
        current_len = len(self.buffer)
        buffer_np = np.array(self.buffer, dtype=np.float32)
        
        # Динамічний паддінг, якщо викид почався менше ніж за 2 години
        if current_len < self.total_steps:
            pad_len = self.total_steps - current_len
            pad_array = np.zeros((pad_len, 8), dtype=np.float32)
            
            # Базова линія датчиків
            pad_array[:, 0:4] = 512.0
            
            # Екстраполяція вітру у минуле для перших розрахунків
            pad_array[:, 4] = buffer_np[0, 4] 
            pad_array[:, 5] = buffer_np[0, 5] 
            pad_array[:, 6] = buffer_np[0, 6] 
            
            full_buffer = np.vstack((pad_array, buffer_np))
        else:
            full_buffer = buffer_np
            
        # Нормалізована часова шкала від 0 до 1
        full_buffer[:, 7] = np.linspace(0, 1, self.total_steps)
        
        # Нормалізація та прогон через модель
        features_scaled = self.feature_scaler.transform(full_buffer)
        x_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(x_tensor)
            
        pred_np = pred_scaled.cpu().numpy()
        pred_real = self.target_scaler.inverse_transform(pred_np)[0]
        
        params = np.zeros(7)
        params[0] = 10 ** pred_real[0] # D0
        params[1] = 10 ** pred_real[1] # Q0
        params[2:7] = pred_real[2:7]   # x0, y0, z0, t0, beta
        
        return params, full_buffer
