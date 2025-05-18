import numpy as np

# 物理定数
k_B = 1.38064852e-23  # ボルツマン定数 (J/K)

def R_opt(e_n, i_n):
    """
    電圧ノイズと電流ノイズの寄与が等しくなる最適な出力抵抗 R_opt (Ω) を計算
    
    パラメータ:
      e_n : float
        入力電圧ノイズ密度 (V/√Hz)
      i_n : float
        入力電流ノイズ密度 (A/√Hz)
    戻り値:
      R_opt : float
        オーム単位の最適抵抗値
    """
    return e_n / i_n

def noise_densities(e_n, i_n, R, T=300):
    """
    各ノイズ密度成分（単位：V/√Hz）を計算:
      - 電圧ノイズ
      - 電流ノイズ（電圧に変換）
      - 熱ノイズ（ジョンソンノイズ）
      - 合計ノイズ密度
    
    パラメータ:
      e_n : float
        入力電圧ノイズ密度 (V/√Hz)
      i_n : float
        入力電流ノイズ密度 (A/√Hz)
      R : float
        出力抵抗 (Ω)
      T : float
        温度（K）, デフォルト 300K
    
    戻り値:
      (e_n, i_n*R, thermal, total) : floatのタプル
    """
    v_noise = e_n
    i_noise = i_n * R
    thermal = np.sqrt(4 * k_B * T * R)
    total = np.sqrt(v_noise**2 + i_noise**2 + thermal**2)
    return v_noise, i_noise, thermal, total

def rms_noise(e_n, i_n, R, bandwidth, T=300):
    """
    与えられた帯域幅でのRMSノイズ電圧（V_rms）を計算
    
    パラメータ:
      e_n : float
        入力電圧ノイズ密度 (V/√Hz)
      i_n : float
        入力電流ノイズ密度 (A/√Hz)
      R : float
        出力抵抗 (Ω)
      bandwidth : float
        帯域幅 (Hz)
      T : float
        温度（K）, デフォルト 300K
    
    戻り値:
      v_rms : float
        帯域幅全体でのRMSノイズ電圧 (V)
    """
    _, _, _, total_density = noise_densities(e_n, i_n, R, T)
    return total_density * np.sqrt(bandwidth)

if __name__ == "__main__":
    # OPA1612の例
    e_n = 1.1e-9    # 1.1 nV/√Hz
    i_n = 1.7e-12   # 1.7 pA/√Hz
    T = 300         # 300 K
    bandwidth = 20e3  # 20 kHz
    
    Ropt = R_opt(e_n, i_n)
    v_n, i_n_volt, th_n, total_nd = noise_densities(e_n, i_n, Ropt, T)
    v_rms = rms_noise(e_n, i_n, Ropt, bandwidth, T)
    
    print(f"最適な抵抗値 R_opt = {Ropt:.1f} Ω")
    print(f"R_optでのノイズ密度:")
    print(f"  電圧ノイズ:        {v_n*1e9:.2f} nV/√Hz")
    print(f"  電流ノイズ:        {i_n_volt*1e9:.2f} nV/√Hz")
    print(f"  熱ノイズ (R):    {th_n*1e9:.2f} nV/√Hz")
    print(f"  合計ノイズ密度:  {total_nd*1e9:.2f} nV/√Hz")
    print(f"{bandwidth/1e3:.0f} Hz帯域でのRMSノイズ: {v_rms*1e6:.2f} μV rms")
