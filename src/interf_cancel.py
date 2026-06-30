import numpy as np
from numpy.linalg import inv


def kalman_filter(y, A, B, sigma_u, sigma_v):
    """
    Filtro de Kalman para el sistema lineal.

    Modelo:
        x[k+1] = A * x[k] + u[k],    u[k] ~ N(0, sigma_u^2 * I)
        y[k]   = B * x[k] + v[k],    v[k] ~ N(0, sigma_v^2 * I)

    Parameters
    ----------
    y : np.ndarray, shape (2, T)
        Observaciones.
    A : np.ndarray, shape (2, 2)
        Matriz de transición de estado.
    B : np.ndarray, shape (2, 2)
        Matriz de observación.
    sigma_u : float
        Desviación estándar del ruido de proceso.
    sigma_v : float
        Desviación estándar del ruido de observación.

    Returns
    -------
    x_est : np.ndarray, shape (2, T+1)
        Estimaciones filtradas del estado. x_est[:, 0] es la estimación
        inicial y x_est[:, t] para t >= 1 es la estimación filtrada
        tras incorporar la observación y[:, t-1].
    P_hist : list
        Lista de matrices de covarianza filtradas.
    """
    d_x = A.shape[0]
    T = y.shape[1]

    Q = (sigma_u ** 2) * np.eye(d_x)   # Covarianza ruido de sistema
    R = (sigma_v ** 2) * np.eye(d_x)   # Covarianza ruido de observación

    # Inicialización con la prior
    x_filt = np.zeros(d_x)
    P_filt = 25.0 * np.eye(d_x)

    x_est = np.zeros((d_x, T + 1))
    P_hist = [None] * (T + 1)
    x_est[:, 0] = x_filt
    P_hist[0] = P_filt.copy()

    for t in range(1, T + 1):
        # --- Predicción ---
        x_pred = A @ x_filt
        P_pred = A @ P_filt @ A.T + Q

        # --- Actualización ---
        S = B @ P_pred @ B.T + R                      # Covarianza de innovación
        K = P_pred @ B.T @ inv(S)                     # Ganancia de Kalman
        innov = y[:, t - 1] - B @ x_pred              # Innovación
        x_filt = x_pred + K @ innov
        P_filt = (np.eye(d_x) - K @ B) @ P_pred

        x_est[:, t] = x_filt
        P_hist[t] = P_filt.copy()

    return x_est, P_hist


def interf_cancel(observations_file, data_rx_file):
    """
    Estima y cancela la interferencia utilizando el filtro de Kalman,
    decodifica la secuencia de bits y la convierte en ASCII.

    El sistema de interferencia es:
        x[k+1] = A * x[k] + u[k]
        y[k]   = B * x[k] + v[k]

    con x[0] = [0, 0] y u[k], v[k] ~ N(0, 0.15^2 * I_2).

    La señal recibida es r[k] = s[k] + x[k], donde s[k] contiene los
    dos bits codificados (como ±1) del bit de información k.

    Parameters
    ----------
    observations_file : str
        Ruta al archivo observations.txt con las observaciones y[k].
    data_rx_file : str
        Ruta al archivo data_rx.txt con los valores recibidos r[k].

    Returns
    -------
    message : str
        Mensaje ASCII decodificado.
    decoded_bits : np.ndarray
        Bits de información decodificados.
    """
    from viterbi_decode_101_111 import viterbi_decode_101_111
    from convert2char import convert2char

    # ---- Parámetros del sistema ----
    A = np.array([[0.9, 0.12],
                  [0.08, 0.85]])

    B = np.array([[-0.12, 0.15],
                  [0.4, -0.15]])

    sigma_u = 0.15
    sigma_v = 0.15

    # ---- Cargar datos ----
    obs_raw = np.loadtxt(observations_file)
    rx_raw = np.loadtxt(data_rx_file)

    # Reshape: cada vector tiene dimensión 2, concatenados en un solo vector
    # [y1[0], y2[0], y1[1], y2[1], ...] -> shape (2, T)
    T_obs = len(obs_raw) // 2
    T_rx = len(rx_raw) // 2

    observations = obs_raw.reshape(-1, 2).T  # shape (2, T_obs)
    data_rx = rx_raw.reshape(-1, 2).T        # shape (2, T_rx)

    # ---- Ejecutar filtro de Kalman para estimar la interferencia ----
    x_est, _ = kalman_filter(observations, A, B, sigma_u, sigma_v)

    # ---- Cancelar interferencia ----
    # x_est[:, 0] es la prior (x[0]=[0,0]), x_est[:, t] estima x[t]
    # Alinear: r[k] = s[k] + x[k], k = 0, ..., T_rx-1
    # Usamos x_est[:, 1:T_rx+1] que son las estimaciones filtradas
    T = min(T_rx, x_est.shape[1] - 1)
    x_interf = x_est[:, 1:T + 1]

    s_clean = data_rx[:, :T] - x_interf

    # ---- Decisión dura sobre los símbolos limpios ----
    # BPSK: +1 -> bit 0, -1 -> bit 1
    coded_bits = (s_clean >= 0).astype(int)

    # Intercalar las dos filas: [c1[0], c2[0], c1[1], c2[1], ...]
    coded_seq = np.zeros(2 * T, dtype=int)
    coded_seq[0::2] = coded_bits[0, :]
    coded_seq[1::2] = coded_bits[1, :]

    # ---- Decodificar con Viterbi ----
    decoded_bits = viterbi_decode_101_111(coded_seq)

    # ---- Eliminar bits de cola (tail bits del codificador) ----
    # El codificador con K=3 añade K-1=2 bits de cola
    # Los eliminamos solo si los últimos bits son ceros de terminación
    n_info = len(decoded_bits) - 2
    if n_info > 0:
        info_bits = decoded_bits[:n_info]
    else:
        info_bits = decoded_bits

    # Ajustar longitud a múltiplo de 8 para conversión ASCII
    n_chars = len(info_bits) // 8
    info_bits_trimmed = info_bits[:n_chars * 8]

    # ---- Convertir a ASCII ----
    message = convert2char(info_bits_trimmed)

    return message, decoded_bits
