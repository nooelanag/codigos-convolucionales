import numpy as np

def conv_encoding(info_bits, gen_matrix):
    """
    Codificador convolucional.

    Codifica una secuencia de bits de información utilizando un código
    convolucional definido por la matriz generadora en forma polinomial.

    Parameters
    ----------
    info_bits : array-like of 0/1
        Secuencia de bits de información a codificar.
    gen_matrix : list of array-like
        Matriz generadora. Cada elemento es un vector binario que representa
        un polinomio generador. Por ejemplo, D^2 + 1 se representa como
        [1, 0, 1] y D^2 + D + 1 como [1, 1, 1].

    Returns
    -------
    coded_bits : np.ndarray
        Secuencia binaria codificada (bits interleaved de cada salida).
    """
    info_bits = np.asarray(info_bits, dtype=int).flatten()
    n_outputs = len(gen_matrix)

    # Convolución de los bits de entrada con cada polinomio generador (mod 2)
    conv_outputs = []
    for g in gen_matrix:
        g = np.asarray(g, dtype=int)
        c = np.convolve(info_bits, g) % 2
        conv_outputs.append(c.astype(int))

    # Todas las salidas tienen la misma longitud: len(info_bits) + K - 1
    # donde K es la longitud de restricción (longitud del polinomio más largo)
    output_len = len(conv_outputs[0])

    # Intercalar (multiplex) las salidas: [c1[0], c2[0], c1[1], c2[1], ...]
    coded_bits = np.zeros(n_outputs * output_len, dtype=int)
    for i, c in enumerate(conv_outputs):
        coded_bits[i::n_outputs] = c

    return coded_bits