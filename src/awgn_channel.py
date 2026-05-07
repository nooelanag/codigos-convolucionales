import numpy as np

def awgn_channel(bits, EbN0_dB, R=1.0):
    """
    Simula la transmisión a través de un canal AWGN con modulación BPSK.

    Mapea los bits a símbolos BPSK (0 -> +1, 1 -> -1), añade ruido
    Gaussiano y devuelve los valores recibidos (soft) y la decisión dura
    (hard decision, bits).

    La varianza del ruido se calcula teniendo en cuenta la tasa del código R:
        sigma^2 = 1 / (2 * R * Eb/N0_lineal)

    De esta forma:
        - Sin codificación (R=1): Pe = Q(sqrt(2 * Eb/N0))
        - Con codificación tasa R: Pe = Q(sqrt(2 * R * Eb/N0))

    Parameters
    ----------
    bits : array-like of 0/1
        Secuencia de bits a transmitir.
    EbN0_dB : float
        Relación señal a ruido por bit de información en dB (Eb/N0).
    R : float, optional
        Tasa del código (por defecto 1.0, sin codificación).

    Returns
    -------
    rx_soft : np.ndarray
        Valores recibidos (señal + ruido), antes de la decisión dura.
    rx_hard : np.ndarray
        Bits recibidos tras decisión dura (umbral en 0).
    """
    bits = np.asarray(bits, dtype=int).flatten()

    # Mapeo BPSK: 0 -> -1, 1 -> +1
    symbols = 2 * bits - 1

    # Conversión de Eb/N0 de dB a lineal
    EbN0_lin = 10.0 ** (EbN0_dB / 10.0)

    # Varianza del ruido: sigma^2 = 1 / (2 * R * Eb/N0)
    noise_var = 1.0 / (2.0 * R * EbN0_lin)
    sigma = np.sqrt(noise_var)

    # Generación de ruido Gaussiano
    noise = sigma * np.random.randn(len(symbols))

    # Señal recibida
    rx_soft = symbols + noise

    # Decisión dura: si rx >= 0 -> bit 1, si rx < 0 -> bit 0
    rx_hard = (rx_soft >= 0).astype(int)

    return rx_soft, rx_hard