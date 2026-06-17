import numpy as np
import matplotlib.pyplot as plt
from generate import generate_sample, direct_model_fractional, DT_MODEL, WINDOW_HOURS, TOTAL_STEPS
from inference import RealTimeInferenceNode

def run_forward_validation(inferred_params, buffer_features):
    D0, Q0, x0, y0, z0, t0, beta = inferred_params
    sig_x, sig_y, sig_z = 1.5, 1.5, 0.5
    
    v_buffer = buffer_features[:, 4]
    sin_t = buffer_features[:, 5]
    cos_t = buffer_features[:, 6]
    theta_buffer = np.arctan2(sin_t, cos_t)
    
    history_steps = int(np.ceil(-t0 / DT_MODEL)) if t0 < 0 else 0
    if history_steps > 0:
        v_hist = np.full(history_steps, v_buffer[0])
        theta_hist = np.full(history_steps, theta_buffer[0])
        v_full = np.concatenate([v_hist, v_buffer])
        theta_full = np.concatenate([theta_hist, theta_buffer])
    else:
        v_full = v_buffer
        theta_full = theta_buffer
        
    pred_concentrations = direct_model_fractional(
        inferred_params, sig_x, sig_y, sig_z, v_full, theta_full
    )
    
    return pred_concentrations


def main():
    print("Генерація тестового потоку даних...")
    features_true, params_true_log = generate_sample(0)
    
    params_true = params_true_log.copy()
    params_true[0] = 10 ** params_true[0]
    params_true[1] = 10 ** params_true[1]
    
    node = RealTimeInferenceNode()
    
    final_pred_params = None
    final_padded_buffer = None
    inference_count = 0
    
    # Списки для збереження історії передбачень параметрів
    history_params = []
    history_times = []  # Час інференсу у хвилинах
    
    print("\nСИМУЛЯЦІЯ РЕАЛЬНОЇ РОБОТИ (2 ГОДИНИ) ---")
    for step in range(TOTAL_STEPS):
        step_data = features_true[step] 
        
        # Подаємо дані на інференс шаг за шагом (имитация реалтайма)
        result = node.process_new_step(step_data)
        
        if result is not None:
            inference_count += 1
            pred_params, padded_buffer = result
            
            # Зберігаємо результати
            final_pred_params = pred_params
            final_padded_buffer = padded_buffer
            
            # Додаємо поточні параметри до історії для другого графіка
            history_params.append(pred_params.copy())
            history_times.append((step * DT_MODEL) / 60.0) # Час у хвилинах
            
            print(f"[{step * DT_MODEL:04d} сек] Інференс #{inference_count} виконано. (D0={pred_params[0]:.2f}, Q0={pred_params[1]:.2f})")
            
    if final_pred_params is None:
        print("\nЗа 2 години газу не було. Інференс не запускався.")
        return
        
    print("\nРЕЗУЛЬТАТИ ОСТАННЬОГО ІНФЕРЕНСУ")
    param_names = ['D0', 'Q0', 'x0 (m)', 'y0 (m)', 'z0 (m)', 't0 (s)', 'beta']
    for name, t_val, p_val in zip(param_names, params_true, final_pred_params):
        print(f"{name:>10}: Істина = {t_val:>8.2f} | Передбачення = {p_val:>8.2f}")
        
    print("\nВалідація за прямою моделлю...")
    pred_concentrations = run_forward_validation(final_pred_params, final_padded_buffer)
    
    # ---------------------------------------------------------
    # 1. ГРАФІК ФІТУВАННЯ КОНЦЕНТРАЦІЙ (СЕНСОРИ)
    # ---------------------------------------------------------
    time_axis = np.linspace(0, WINDOW_HOURS * 60, TOTAL_STEPS)
    true_concentrations = features_true[:, :4]
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Динаміка: Істинні дані сенсорів vs Реконструкція мережі (Останній інференс)', fontsize=15, y=0.96)
    
    for i, ax in enumerate(axes1.flatten()):
        ax.plot(time_axis, true_concentrations[:, i], label='Ground Truth (Сенсор)', color='royalblue', linewidth=2)
        ax.plot(time_axis, pred_concentrations[:, i], label='Prediction (Пряма модель)', color='darkorange', linestyle='--', linewidth=2)
        ax.axhline(512.0, color='gray', linestyle=':', label='Базова лінія')
        ax.set_title(f'Сенсор {i+1}')
        ax.set_xlabel('Час (хвилини)')
        ax.set_ylabel('Сигнал АЦП (у.о.)')
        ax.set_ylim(0, 1050)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    fig1.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)
    plt.savefig('inference_fit_result.png', dpi=300)
    
    # ---------------------------------------------------------
    # 2. ГРАФІКИ ДИНАМІКИ ПЕРЕДБАЧЕННЯ ПАРАМЕТРІВ
    # ---------------------------------------------------------
    history_params = np.array(history_params)
    
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 16))
    fig2.suptitle('Динаміка передбачення параметрів джерела нейромережею', fontsize=16, y=0.96)
    
    axes_flat = axes2.flatten()
    
    for k, name in enumerate(param_names):
        ax = axes_flat[k]
        # Лінія передбачень
        ax.plot(history_times, history_params[:, k], marker='o', markersize=4, color='darkorange', linewidth=1.5, label='Передбачення')
        # Лінія істини
        ax.axhline(params_true[k], color='royalblue', linestyle='--', linewidth=2, label='Істина (Ground Truth)')
        
        ax.set_title(f'Параметр: {name}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Час симуляції (хвилини)', fontsize=10)
        ax.set_ylabel('Значення', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
    # Приховуємо останній (восьмий) порожній графік
    axes_flat[-1].axis('off')
    
    fig2.subplots_adjust(top=0.90, hspace=0.45, wspace=0.25)
    
    plt.savefig('inference_parameters_tracking.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
