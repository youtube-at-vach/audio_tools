import math

def compute_noise(R1, V1, R2, V2, T=300):
    """
    Compute e_n (voltage noise density) and i_n (current noise density)
    from two measurements.

    Parameters:
    R1, R2: input resistances in ohms
    V1, V2: measured noise densities in V/√Hz
    T: temperature in Kelvin (default: 300K)

    Returns:
    (e_n, i_n): tuple of noise densities (V/√Hz, A/√Hz)
    """
    k = 1.380649e-23  # Boltzmann constant (J/K)

    # Convert to squared terms
    V1_sq = V1**2
    V2_sq = V2**2

    # Thermal noise voltages squared
    eR1_sq = 4 * k * T * R1
    eR2_sq = 4 * k * T * R2

    # Solve for i_n^2 using subtraction
    numerator = V2_sq - V1_sq - (eR2_sq - eR1_sq)
    denominator = R2**2 - R1**2
    i_n_sq = numerator / denominator
    if i_n_sq < 0:
        raise ValueError("Computed negative i_n squared; check input values.")
    i_n = math.sqrt(i_n_sq)

    # Solve for e_n^2 from first equation
    e_n_sq = V1_sq - (i_n * R1)**2 - eR1_sq
    if e_n_sq < 0:
        raise ValueError("Computed negative e_n squared; check input values.")
    e_n = math.sqrt(e_n_sq)

    return e_n, i_n

def main():
    print("Compute input-referred voltage (e_n) and current (i_n) noise densities\n")
    R1 = float(input("Enter R1 (Ω): "))
    V1 = float(input("Enter V1 (V/√Hz): "))
    R2 = float(input("Enter R2 (Ω): "))
    V2 = float(input("Enter V2 (V/√Hz): "))
    T = input("Enter temperature in K [default 300]: ")
    T = float(T) if T else 300.0

    try:
        e_n, i_n = compute_noise(R1, V1, R2, V2, T)
        print(f"\ne_n (voltage noise density): {e_n:.3e} V/√Hz")
        print(f"i_n (current noise density): {i_n:.3e} A/√Hz")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

