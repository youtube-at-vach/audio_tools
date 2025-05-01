import numpy as np

# Physical constant
k_B = 1.38064852e-23  # Boltzmann constant in J/K

def R_opt(e_n, i_n):
    """
    Calculate optimal source resistance R_opt (Ω) 
    where voltage-noise and current-noise contributions are equal.
    
    Parameters:
      e_n : float
        Input voltage noise density (V/√Hz)
      i_n : float
        Input current noise density (A/√Hz)
    Returns:
      R_opt : float
        Optimal resistance in ohms
    """
    return e_n / i_n

def noise_densities(e_n, i_n, R, T=300):
    """
    Calculate individual noise density contributions (in V/√Hz):
      - voltage noise
      - current noise (converted to voltage)
      - thermal (Johnson) noise
      - total noise density
    
    Parameters:
      e_n : float
        Input voltage noise density (V/√Hz)
      i_n : float
        Input current noise density (A/√Hz)
      R : float
        Source resistance (Ω)
      T : float
        Temperature (K), default 300K
    
    Returns:
      (e_n, i_n*R, thermal, total) : tuple of floats
    """
    v_noise = e_n
    i_noise = i_n * R
    thermal = np.sqrt(4 * k_B * T * R)
    total = np.sqrt(v_noise**2 + i_noise**2 + thermal**2)
    return v_noise, i_noise, thermal, total

def rms_noise(e_n, i_n, R, bandwidth, T=300):
    """
    Calculate RMS noise voltage (V_rms) over a given bandwidth.
    
    Parameters:
      e_n : float
        Input voltage noise density (V/√Hz)
      i_n : float
        Input current noise density (A/√Hz)
      R : float
        Source resistance (Ω)
      bandwidth : float
        Bandwidth (Hz)
      T : float
        Temperature (K), default 300K
    
    Returns:
      v_rms : float
        RMS noise voltage over the bandwidth (V)
    """
    _, _, _, total_density = noise_densities(e_n, i_n, R, T)
    return total_density * np.sqrt(bandwidth)

if __name__ == "__main__":
    # Example for OPA1612
    e_n = 1.1e-9    # 1.1 nV/√Hz
    i_n = 1.7e-12   # 1.7 pA/√Hz
    T = 300         # 300 K
    bandwidth = 20e3  # 20 kHz
    
    Ropt = R_opt(e_n, i_n)
    v_n, i_n_volt, th_n, total_nd = noise_densities(e_n, i_n, Ropt, T)
    v_rms = rms_noise(e_n, i_n, Ropt, bandwidth, T)
    
    print(f"Optimal Resistance R_opt = {Ropt:.1f} Ω")
    print(f"Noise densities at R_opt:")
    print(f"  Voltage noise:        {v_n*1e9:.2f} nV/√Hz")
    print(f"  Current noise:        {i_n_volt*1e9:.2f} nV/√Hz")
    print(f"  Thermal noise (R):    {th_n*1e9:.2f} nV/√Hz")
    print(f"  Total noise density:  {total_nd*1e9:.2f} nV/√Hz")
    print(f"RMS noise over {bandwidth/1e3:.0f} Hz: {v_rms*1e6:.2f} μV rms")
