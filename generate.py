# ====================================================================================== #
# МОДУЛЬ ГЕНЕРАЦІЇ ДАТАСЕТУ ДЛЯ ТРЕНУВАННЯ НЕЙРОМЕРЕЖЕВОГО РОЗВ'ЯЗУВАЧА ОБЕРНЕНОЇ ЗАДАЧІ #
# ====================================================================================== #
import os
import numpy as np
import h5py
import multiprocessing as mp
from scipy.optimize import nnls
from scipy.interpolate import interp1d
from scipy.special import gamma
from numba import njit

# ==========================================
# 1. КОНСТАНТИ ТА НАЛАШТУВАННЯ (Дах автомобіля)
# ==========================================
NUM_SAMPLES = 100000        # Кількість зразків
WINDOW_HOURS = 2         
DT_MODEL = 10            
TOTAL_STEPS = int((WINDOW_HOURS * 3600) / DT_MODEL) # 720 кроків
BETA_GRID = np.linspace(0.1, 1.0, 500) # Сітка з 500 значень beta від 0.1 до 1.0 для look-up table

CAR_ROOF_HEIGHT = 1.5     
R_tetra = 0.19            
Z_base = CAR_ROOF_HEIGHT + 0.1  
Z_apex = CAR_ROOF_HEIGHT + 0.3  

SENSORS = np.array([
    [R_tetra * np.cos(0),           R_tetra * np.sin(0),           Z_base], 
    [R_tetra * np.cos(2 * np.pi/3), R_tetra * np.sin(2 * np.pi/3), Z_base], 
    [R_tetra * np.cos(4 * np.pi/3), R_tetra * np.sin(4 * np.pi/3), Z_base], 
    [0.0,                           0.0,                           Z_apex]  
])

# ==========================================
# 2. ОБЧИСЛЕННЯ ВАГОВИХ КОЕФІЦІЄНТІВ ДРОБОВОГО ЯДРА (NNLS)
# ==========================================
def compute_fractional_weights(T_max_val, dt, N_nodes=30, M_steps=200):
    # Вузли s залежать лише від T_max, не від beta
    j = np.arange(1, N_nodes + 1)
    s = (1.0 / T_max_val) * (T_max_val / dt) ** ((j - 1) / (N_nodes - 1))
    
    k = np.arange(1, M_steps + 1)
    t = dt * (T_max_val / dt) ** ((k - 1) / (M_steps - 1))
    A = np.exp(-s[None, :] * t[:, None])
    
    w_grid = np.zeros((len(BETA_GRID), N_nodes))
    for i, b_val in enumerate(BETA_GRID):
        b = (t ** (b_val - 1.0)) / gamma(b_val)
        w, _ = nnls(A, b)
        w_grid[i] = w
        
    return s, w_grid

T_MAX_EXT = 2 * (WINDOW_HOURS * 3600)
LUT_FILE = "beta_grid_lut.npz"

if not os.path.exists(LUT_FILE):
    # Якщо файл таблиці відсутній - розраховуємо
    print("Попереднє обчислення сітки ваг методу суми експонент...")
    S_EXT, W_GRID_EXT = compute_fractional_weights(T_MAX_EXT, DT_MODEL)
    np.savez(LUT_FILE, S_EXT=S_EXT, W_GRID_EXT=W_GRID_EXT)

# Завантажуємо коефіцієнти з таблиці
lut = np.load(LUT_FILE)
S_EXT, W_GRID_EXT = lut["S_EXT"], lut["W_GRID_EXT"]

# Об'єкти для швидкої лінійної інтерполяції
interp_w_ext = interp1d(BETA_GRID, W_GRID_EXT, axis=0, fill_value="extrapolate")

# ==========================================
# 3. КУСКОВО-СТАЛИЙ ВІТЕР
# ==========================================
def generate_wind(steps):
    v = np.zeros(steps)
    theta = np.zeros(steps)
    current_step = 0
    while current_step < steps:
        duration = np.random.randint(18, 73)
        end_step = min(current_step + duration, steps)
        v[current_step:end_step] = np.random.uniform(0.5, 10.0)
        theta[current_step:end_step] = np.random.uniform(0, 2 * np.pi)
        current_step = end_step
    return v, theta

# ==========================================
# 4. ШВИДКЕ ОБЧИСЛЕННЯ ПРЯМОЇ МОДЕЛІ (NUMBA NJIT)
# ==========================================
@njit(fastmath=True, cache=True)
def run_time_stepping(time_array, t0, history_steps, v, theta, D0, C_den, 
                      Omega_x, Omega_y, Omega_z, Omega_sq_sum, q_tilde_shape, 
                      exp_phases, d_omega_vol,
                      exp_s_dt_expanded, coef1_expanded, coef2_expanded, w_expanded,
                      N_nodes, Nx, Ny, Nz, total_steps):
    
    # Ініціалізація історії H_j і q_tilde (виділення пам'яті лише 1 раз)
    q_tilde = q_tilde_shape.astype(np.complex128)
    H = np.zeros((N_nodes, Nx, Ny, Nz), dtype=np.complex128)
    concentrations = np.zeros((total_steps, 4))
    
    num_sum = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    fourier_factor = (1.0 / (2.0 * np.pi)**3) * d_omega_vol

    # Часовий крок рекурсії
    for n in range(len(time_array)):
        t_curr = time_array[n]
        
        # Модель активна лише якщо нинішній час більше часу викиду t0
        if t_curr <= t0:
            continue
            
        vn = v[n]
        thetan = theta[n]
        sin_t = np.sin(thetan)
        cos_t = np.cos(thetan)
        
        # Очиска буфера чисельника
        num_sum.fill(0.0 + 0.0j)
        
        # Явне швидке обчислення чисельника
        for j in range(N_nodes):
            w_j = w_expanded[j, 0, 0, 0]
            e_s = exp_s_dt_expanded[j, 0, 0, 0]
            c1  = coef1_expanded[j, 0, 0, 0]
            
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        # Константа Lambda_n
                        lam = (1j * (vn * sin_t)) * Omega_x[ix, iy, iz] + \
                              (1j * (vn * cos_t)) * Omega_y[ix, iy, iz] - \
                              (D0 * Omega_sq_sum[ix, iy, iz])
                        
                        term1 = e_s * H[j, ix, iy, iz]
                        term2 = lam * c1 * q_tilde[ix, iy, iz]
                        num_sum[ix, iy, iz] += w_j * (term1 + term2)
        
        # Оновлення спектральної концентрації q_tilde та шарів історії H_j
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    lam = (1j * (vn * sin_t)) * Omega_x[ix, iy, iz] + \
                          (1j * (vn * cos_t)) * Omega_y[ix, iy, iz] - \
                          (D0 * Omega_sq_sum[ix, iy, iz])
                    
                    # Знаменник рівняння
                    den = 1.0 - lam * C_den
                    q_tilde_new = (q_tilde_shape[ix, iy, iz] + num_sum[ix, iy, iz]) / den
                    
                    # Оновлення історії H_j для наступного кроку
                    for j in range(N_nodes):
                        e_s = exp_s_dt_expanded[j, 0, 0, 0]
                        c1  = coef1_expanded[j, 0, 0, 0]
                        c2  = coef2_expanded[j, 0, 0, 0]
                        
                        term1 = e_s * H[j, ix, iy, iz]
                        q_term = c1 * q_tilde[ix, iy, iz] + c2 * q_tilde_new
                        H[j, ix, iy, iz] = term1 + lam * q_term
                        
                    q_tilde[ix, iy, iz] = q_tilde_new

        # Обернене перетворення Фур'є в точках розміщення 4-х сенсорів
        # Обчислюємо фізичну концентрацію лише для основного вікна вимірювання (t >= 0)
        if t_curr >= 0.0:
            idx_concentrations = n - history_steps
            
            for s_idx in range(4):
                val = 0.0 + 0.0j
                for ix in range(Nx):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            val += q_tilde[ix, iy, iz] * exp_phases[s_idx, ix, iy, iz]
                
                # Чисельне інтегрування по об'єму частот
                val *= fourier_factor
                C_phys = val.real if val.real > 0.0 else 0.0
                
                # Цифровий двійник MiCS-5524
                alpha = 0.65 
                Rs_R0 = 1.0 / (1.0 + 0.5 * C_phys)**alpha
                
                # Переведення в напругу та імітація 10-бітного АЦП Arduino
                V_max = 5.0  
                V_out = V_max * (1.0 / (1.0 + Rs_R0))
                concentrations[idx_concentrations, s_idx] = (V_out / V_max) * 1023.0
                
    return concentrations

# ==========================================
# 5. ТОЧНА ПРЯМА МОДЕЛЬ (ОБГОРТКА)
# ==========================================
def direct_model_fractional(params, sig_x, sig_y, sig_z, v, theta):
    D0, Q0, x0, y0, z0, t0, beta = params
    dt = DT_MODEL
    T_max = WINDOW_HOURS * 3600
    
    # Знаходження ваг w_j та вузлів s_j із оптимізаційної задачі
    s = S_EXT
    w = interp_w_ext(beta)
        
    N_nodes = len(w)
    
    # Створення тривимірної сітки частот
    Nx, Ny, Nz = 15, 15, 11
    wx = np.linspace(-6.0, 6.0, Nx)
    wy = np.linspace(-6.0, 6.0, Ny)
    wz = np.linspace(-6.0, 6.0, Nz)
    
    d_omega_vol = (wx[1] - wx[0]) * (wy[1] - wy[0]) * (wz[1] - wz[0])
    Omega_x, Omega_y, Omega_z = np.meshgrid(wx, wy, wz, indexing='ij')
    
    # Обчислення константних масивів перед початком циклу
    Omega_sq_sum = Omega_x**2 + Omega_y**2 + Omega_z**2
    source_shift = Omega_x * x0 + Omega_y * y0
    
    # Комплексні екпоненти для усіх 4-х сенсорів (Форма: 4, 15, 15, 11)
    exp_phases = np.array([
        np.exp(1j * (Omega_x * sx + Omega_y * sy + Omega_z * sz - source_shift)) 
        for sx, sy, sz in SENSORS
    ], dtype=np.complex128)

    # Обчислення тимчасових коефіцієнтів для рекурентних співвідношень
    s_dt = s * dt
    coef1 = (1.0 - (s_dt + 1.0) * np.exp(-s_dt)) / (s**2 * dt)
    coef2 = (np.exp(-s_dt) + s_dt - 1.0) / (s**2 * dt)
    C_den = np.sum(w * coef2)  
    
    # Початкова умова q_tilde_0 
    q_tilde_shape = 2.0 * Q0 * (2.0 * np.pi)**1.5 * sig_x * sig_y * sig_z * \
                    np.cos(Omega_z * z0) * \
                    np.exp(-0.5 * (Omega_x**2 * sig_x**2 + Omega_y**2 * sig_y**2 + Omega_z**2 * sig_z**2))
    
    # Формування розширеної часової шкали
    history_steps = int(np.ceil(-t0 / dt)) if t0 < 0 else 0
    time_pre = 0.0 - np.arange(history_steps, 0, -1) * dt
    time_main = np.arange(0, TOTAL_STEPS) * dt
    time_array = np.concatenate([time_pre, time_main])
    
    # Розширення розмірностей коефіцієнтів для Numba (щоб брати скаляри всередині циклу)
    exp_s_dt_expanded = np.exp(-s_dt)[:, None, None, None]
    coef1_expanded = coef1[:, None, None, None]
    coef2_expanded = coef2[:, None, None, None]
    w_expanded = w[:, None, None, None]
    
    # Запуск JIT-рушія
    concentrations = run_time_stepping(
        time_array, t0, history_steps, v, theta, D0, C_den,
        Omega_x, Omega_y, Omega_z, Omega_sq_sum, q_tilde_shape,
        exp_phases, d_omega_vol,
        exp_s_dt_expanded, coef1_expanded, coef2_expanded, w_expanded,
        N_nodes, Nx, Ny, Nz, TOTAL_STEPS
    )
    
    # Реалістичний шум: +-1.5 кроку АЦП
    noise = np.random.normal(0, 1.5, concentrations.shape)
    concentrations = np.clip(concentrations + noise, 0.0, 1023.0)
    
    return concentrations

# ==========================================
# 6. ГЕНЕРАЦІЯ НОВОГО ЗРАЗКА
# ==========================================
def generate_sample(sample_id):
    # Додаємо цикл: генеруємо доти, доки датчики не побачать реальний газ
    while True:
        beta = np.random.uniform(0.1, 1.0)
        D0 = np.random.uniform(0.5, 8.0)
        Q0 = np.random.uniform(1.0, 500.0)
        
        # Фіксований розмір витоку для усунення локальних мінімумів по Q0
        sig_x = 1.5
        sig_y = 1.5
        sig_z = 0.5
        
        x0 = np.random.uniform(-3.0, 3.0)
        y0 = np.random.uniform(-3.0, 3.0)
        z0 = np.random.uniform(0.0, 2.5)
        
        # t0 обмежено вікном моделювання (в секундах) — від -2 год до 0 год (момент пробудження)
        t0 = np.random.uniform(-WINDOW_HOURS * 3600, 0.0)
        
        params = np.array([D0, Q0, x0, y0, z0, t0, beta])
        
        # Обчислюємо загальне число кроків для генерації профілю вітру з урахуванням передісторії
        history_steps = int(np.ceil(-t0 / DT_MODEL)) if t0 < 0 else 0
        total_sim_steps = history_steps + TOTAL_STEPS
        
        v, theta = generate_wind(total_sim_steps)
        concentrations = direct_model_fractional(params, sig_x, sig_y, sig_z, v, theta)
        
        # Фільтр "мертвих" даних: якщо максимальний сигнал не перевищує 520.0 кроків АЦП (базова лінія датчика + шум за правилом трьох сигм), значить хмара пройшла повз. Перегенеруємо!
        if np.max(concentrations) > 520.0:
            break # Сигнал є, виходимо з циклу і продовжуємо обробку
            
    time_array = np.linspace(0, WINDOW_HOURS * 3600, TOTAL_STEPS)
    t_norm = time_array / (WINDOW_HOURS * 3600)
    
    # Зрізаємо масиви вітру, залишаючи для ознак лише період спостереження (t >= 0)
    theta_cut = theta[history_steps:]
    features = np.column_stack((concentrations, v[history_steps:], np.sin(theta_cut), np.cos(theta_cut), t_norm))
    
    # Логарифм D0 і Q0 для кращої роботи Transformer
    params[0] = np.log10(params[0])
    params[1] = np.log10(params[1])
    
    return features, params

if __name__ == '__main__':
    # Форсуємо метод запуску процесів spawn для Всіх операційних систем
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # На випадок, якщо spawn уже встановлено в системі
        pass

    print(f"Запуск генерації {NUM_SAMPLES} зразків на базі дробової моделі Фур'є...")
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Активно ядер CPU: {num_cores}")
    
    with h5py.File('fractional_diffusion_dataset.h5', 'w') as h5f:
        X_dset = h5f.create_dataset('features', shape=(NUM_SAMPLES, TOTAL_STEPS, 8), dtype=np.float32)
        y_dset = h5f.create_dataset('targets', shape=(NUM_SAMPLES, 7), dtype=np.float32)
        
        with mp.Pool(num_cores) as pool:
            results = pool.imap_unordered(generate_sample, range(NUM_SAMPLES), chunksize=100)
            
            for i, (features, params) in enumerate(results):
                X_dset[i] = features
                y_dset[i] = params
                
                if (i + 1) % 50 == 0:
                    print(f"Прогрес: {i + 1} / {NUM_SAMPLES} згенеровано.", flush=True)
                    
    print("Датасет успішно збережено у 'fractional_diffusion_dataset.h5'")
