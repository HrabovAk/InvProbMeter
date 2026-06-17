# =============================================================== #
# МОДУЛЬ ТРЕНУВАННЯ НЕЙРОМЕРЕЖЕВОГО РОЗВ'ЯЗУВАЧА ОБЕРНЕНОЇ ЗАДАЧІ #
# =============================================================== #
import os
import random
import pickle
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model import Solver

def set_seed(seed=42):
    #Фіксує всі генератори випадкових чисел для відтворюваності результатів.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Якщо використовується декілька GPU
        
    # Ці два налаштування роблять згортки (Conv1d) детермінованими.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Безпечний Dataset для роботи з HDF5 файлами у multi-processing режимі. Файл відкривається індивідуально всередині кожного воркера під час першого запиту.
# ==========================================
class InitDataset(Dataset):
    def __init__(self, file_path, indices, feature_scaler, target_scaler):
        self.file_path = file_path
        self.indices = indices
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.h5_file = None # Ініціалізуємо ліниво всередині __getitem__

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
            
        real_idx = self.indices[idx]
        x = self.h5_file['features'][real_idx] # Форма: (72, 8)  
        y = self.h5_file['targets'][real_idx]  # Форма: (7,) 

        # Масштабуємо фічі: трансформуємо двовимірну матрицю (72, 8), потім повертаємо форму
        x_scaled = self.feature_scaler.transform(x)

        # Масштабуємо таргети
        y_scaled = self.target_scaler.transform(y.reshape(1, -1))[0]
        
        return (
            torch.tensor(x_scaled, dtype=torch.float32), 
            torch.tensor(y_scaled, dtype=torch.float32)
        )

    def __del__(self):
        # Безпечно закриваємо файл, якщо він був відкритий у цьому процесі
        if self.h5_file is not None:
            self.h5_file.close()

# ==========================================
# 2. ОСНОВНИЙ ПАЙПЛАЙН НАВЧАННЯ НЕЙРОМЕРЕЖІ.
# ==========================================
def main():
    # Фіксація сиду для повної відтворюваності
    set_seed(42)
    
    # Налаштування
    h5_path = 'fractional_diffusion_dataset.h5'
    batch_size = 64
    epochs = 1000
    lr = 2e-4
    val_split = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Використовуємо пристрій: {device}")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Файл {h5_path} не знайдено! Спочатку запустіть generate.py")

    # Розділення на Train/Val підвибірки для запобігання Data Leakage
    with h5py.File(h5_path, 'r') as h5f:
        total_samples = h5f['targets'].shape[0]

    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    val_size = int(total_samples * val_split)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Розрахунок скейлерів без перевантаження RAM (Фіт тільки на тренувальній вибірці)
    print("Розрахунок скейлерів для нормалізації даних...")
    with h5py.File(h5_path, 'r') as h5f:
        # Таргети тренувальні — читаємо для ідеального фіту (обов'язково сортуємо індекси для h5py)
        sorted_train_indices = np.sort(train_indices)
        targets_train = h5f['targets'][sorted_train_indices]
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        target_scaler.fit(targets_train)

        # Фічі великі (близько 1 Гб)
        sorted_train_indices = np.sort(train_indices) # Обов'язкове сортування індексів для h5py
        features_train = h5f['features'][sorted_train_indices] # (N_samples, 72, 8) 
        # Згортаємо часову вісь, щоб рахувати статистику по 8 каналах глобально
        features_train_flattened = features_train.reshape(-1, 8)
        
        # Для фіч використовуємо StandardScaler
        feature_scaler = StandardScaler()
        feature_scaler.fit(features_train_flattened)

    # Зберігаємо скейлери на диск — вони життєво необхідні для інференсу на сервері!
    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, f)
    print("Скейлери успешно збережено у 'scalers.pkl'")
    
    train_dataset = InitDataset(h5_path, train_indices, feature_scaler, target_scaler)
    val_dataset = InitDataset(h5_path, val_indices, feature_scaler, target_scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Ініціалізація Solver та Оптимізатора
    model = Solver(input_dim=8, output_dim=7).to(device)
    
    if os.path.exists('best_solver_weights.pth'):
        # Якщо є попередні ваги - донавчаємо наявну модель
        print("Знайдено збережені ваги 'best_solver_weights.pth'. Завантажуємо для донавчання...")
        model.load_state_dict(torch.load('best_solver_weights.pth', map_location=device))
    else:
        print("Попередніх ваг не знайдено, починаємо з нуля...")
    
    # Емпіричні ваги для параметрів: [D0, Q0, x0, y0, z0, t0, beta]
    loss_weights = torch.tensor([1.2, 1.0, 2.5, 2.5, 1.5, 4.0, 3.0], dtype=torch.float32).to(device)
    
    # Ідеальний лос для фізичної регресії — Huber Loss (стійкий до викидів) з урахуванням власних ваг
    criterion = nn.HuberLoss(delta=0.5, reduction='none')
    
    # AdamW з ваговою регуляризацією, щоб мережа не божеволіла від шуму датчиків
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Динамічне зниження LR, якщо плато затягнулося
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Тренувальний цикл
    best_val_loss = float('inf')
    print(f"Старт навчання. Навчання: {len(train_indices)}, Валідація: {len(val_indices)}")
    
    for epoch in range(1, epochs + 1):
        # Фаза навчання
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch)
            
            # Розрахунок зваженого лоса
            unreduced_loss = criterion(predictions, y_batch)
            weighted_loss = unreduced_loss * loss_weights
            loss = weighted_loss.mean()
            
            loss.backward()

            # Кліпінг градієнтів (захист від вибухів)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        # Фаза валідації
        model.eval()
        val_loss = 0.0
        
        # Масив для накопичення абсолютної похибки (MAE) у фізичних величинах для 7 параметрів
        val_mae_accum = np.zeros(7)
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                
                # Розрахунок зваженого лоса для валідації
                unreduced_loss = criterion(predictions, y_batch)
                weighted_loss = unreduced_loss * loss_weights
                loss = weighted_loss.mean()
                
                val_loss += loss.item() * x_batch.size(0)
                
                # Розрахунок MAE
                # Переводимо тензори назад у numpy (на CPU)
                preds_np = predictions.cpu().numpy()
                targets_np = y_batch.cpu().numpy()
                
                # Зворотне масштабування до реальних фізичних одиниць
                preds_real = target_scaler.inverse_transform(preds_np)
                targets_real = target_scaler.inverse_transform(targets_np)
                
                # Потенціонування D0 і Q0
                preds_real[:, 0] = 10 ** preds_real[:, 0]
                preds_real[:, 1] = 10 ** preds_real[:, 1]
                
                targets_real[:, 0] = 10 ** targets_real[:, 0]
                targets_real[:, 1] = 10 ** targets_real[:, 1]
                
                # Рахуємо абсолютну сумарну похибку для кожного параметра в поточному батчі
                val_mae_accum += np.abs(preds_real - targets_real).sum(axis=0)
                
        val_loss /= len(val_loader.dataset)
        
        # Усереднюємо MAE по всьому валідаційному датасету
        val_mae_real = val_mae_accum / len(val_loader.dataset)

        # Крок скедулера по валідаційному лосу
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Епоха [{epoch}/{epochs}] | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Виводимо фізичну помилку
        print(f"MAE: D0={val_mae_real[0]:.2f}, Q0={val_mae_real[1]:.1f}, ")
        print(f"     x0={val_mae_real[2]:.2f}м, y0={val_mae_real[3]:.2f}м, z0={val_mae_real[4]:.2f}м, ")
        print(f"     t0={val_mae_real[5]/60:.1f}хв, beta={val_mae_real[6]:.3f}")

        # Зберігаємо лише найкращі ваги
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_solver_weights.pth')
            print(f"--> Знайдено найкращу модель! Ваги збережено.")
            
    print("\nГотово!")

if __name__ == '__main__':
    main()
