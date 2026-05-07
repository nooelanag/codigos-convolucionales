import numpy as np

def convert2char(bits):

    bits = np.array(bits)

    # Reshape into rows of 8 bits
    aux = bits.reshape(-1, 8)

    # Convert each row of 8 bits to decimal
    decimals = aux.dot(1 << np.arange(7, -1, -1))

    # Convert decimals to characters
    message = ''.join(chr(d) for d in decimals)

    return message