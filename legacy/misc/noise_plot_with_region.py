import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# SI接頭辞で軸の表示フォーマット関数
def si_formatter(x, pos):
    # 定義：値の範囲と接頭辞
    prefixes = [
        (1e12, 'T'),
        (1e9,  'G'),
        (1e6,  'M'),
        (1e3,  'k'),
        (1,    ''),
        (1e-3, 'm'),
        (1e-6, 'µ'),
        (1e-9, 'n'),
        (1e-12,'p'),
        (1e-15,'f'),
    ]

    abs_x = abs(x)
    for factor, prefix in prefixes:
        if abs_x >= factor:
            value = x / factor
            # 小数第1〜2桁まで表示（整数なら整数表示）
            if value >= 100:
                return f"{value:.0f} {prefix}"
            elif value >= 10:
                return f"{value:.1f} {prefix}"
            else:
                return f"{value:.2g} {prefix}"
    
    return f"{x:.2g}"  # ごく小さい値の場合など

# 定数
k = 1.38e-23  # ボルツマン定数 [J/K]
T = 300       # 温度 [K]

# アンプの雑音パラメータ（例）
v_n = 1.1e-9     # 電圧雑音密度 [V/√Hz]
i_n = 1.7e-12    # 電流雑音密度 [A/√Hz]

# 信号源抵抗の範囲（1Ω～1GΩ）
R = np.logspace(0, 9, 1000)

# 雑音計算
thermal_noise = np.sqrt(4 * k * T * R)     # 熱雑音密度
current_noise = i_n * R                    # 電流雑音の電圧換算
total_noise = np.sqrt(v_n**2 + current_noise**2 + thermal_noise**2)  # 合成雑音密度

# 合成雑音 ≤ √2 × 熱雑音 の範囲を抽出
threshold = np.sqrt(2) * thermal_noise
valid_region = total_noise <= threshold
valid_indices = np.where(valid_region)[0]
R_min = R[valid_indices[0]] if valid_indices.size > 0 else None
R_max = R[valid_indices[-1]] if valid_indices.size > 0 else None

# 雑音最小点の表示
R_opt = v_n / i_n

plt.figure(figsize=(10, 6))
plt.loglog(R, total_noise, label='Total Noise', color='black')
plt.axhline(v_n, linestyle='--', color='red', label=f'Voltage Noise (v_n = {si_formatter(v_n, None)}V/√Hz)')
plt.loglog(R, current_noise, '--', color='green', label=f'Current Noise (i_n = {si_formatter(i_n, None)}A/√Hz × R)')
plt.loglog(R, thermal_noise, '--', color='blue', label='Thermal Noise (√4kTR)')

# ラベル表示位置
x_pos = R[-1]  # 右端（最大抵抗）
# 電圧雑音は水平線なので適当なX軸値に
plt.text(x_pos, v_n*1.1, f'{si_formatter(v_n, None)}V/√Hz', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

# # 電流雑音の右端値
y_in = i_n * x_pos
plt.text(x_pos, y_in*1.1, f'{si_formatter(i_n, None)}A/√Hz × R', color='green', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

# 推奨範囲の塗りつぶし
if R_min and R_max:
    plt.axvspan(R_min, R_max, color='orange', alpha=0.2,
                label=f'Preferred Region\n({si_formatter(R_min, None)}Ω – {si_formatter(R_max, None)}Ω)')

# 最適インピーダンス点
plt.axvline(R_opt, color='magenta', linestyle=':', label=f'R_opt = {si_formatter(R_opt, None)}Ω')

plt.xlabel('Source Resistance [Ω]')
plt.ylabel('Voltage Noise Density [V/√Hz]')
plt.title('Input-Referred Noise vs Source Resistance')
plt.grid(True, which='both', ls=':')

# SI単位で軸の数値を表示
plt.gca().xaxis.set_major_formatter(FuncFormatter(si_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(si_formatter))

plt.legend()
plt.tight_layout()
plt.show()
