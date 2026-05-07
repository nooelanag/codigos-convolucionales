import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

from conv_encoding import conv_encoding
from awgn_channel import awgn_channel
from viterbi_decode_101_111 import viterbi_decode_101_111
from interf_cancel import interf_cancel


def Q(x):
    """Función Q: Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def main():
    # ======================================================================
    #  Parámetros de simulación
    # ======================================================================
    # Para obtener estadísticas fiables hasta BER ~ 10^-4, se puede
    # aumentar n_bits y n_frames (a costa de mayor tiempo de ejecución).
    EbN0_dB_range = np.arange(0, 11, 1)        # Eb/N0 en dB
    n_bits = 1000                                # Bits de información por trama
    n_frames = 100                               # Número de tramas por punto

    # Matriz generadora: G(D) = [D^2 + 1, D^2 + D + 1]
    g1 = [1, 0, 1]
    g2 = [1, 1, 1]
    gen_matrix = [g1, g2]
    R = 0.5                                      # Tasa del código

    # Almacenamiento de resultados
    ber_uncoded = np.zeros(len(EbN0_dB_range))
    ber_coded = np.zeros(len(EbN0_dB_range))
    ber_coded_e2e = np.zeros(len(EbN0_dB_range))

    print("=" * 60)
    print("  Simulación BER: sin codificar vs. código convolucional")
    print("  G(D) = [D^2+1, D^2+D+1],  R = 1/2")
    print(f"  {n_bits} bits/trama x {n_frames} tramas = "
          f"{n_bits * n_frames} bits/punto")
    print("=" * 60)

    # ======================================================================
    #  Simulación Monte Carlo
    # ======================================================================
    for idx, EbN0_dB in enumerate(EbN0_dB_range):
        errors_uncoded = 0
        total_uncoded = 0
        errors_coded = 0
        errors_coded_e2e = 0
        total_coded = 0
        total_coded_e2e = 0

        for _ in range(n_frames):
            # --- Generar bits aleatorios ---
            info_bits = np.random.randint(0, 2, n_bits)

            # ==============================================================
            #  Canal sin codificar (R = 1)
            # ==============================================================
            _, rx_uncoded = awgn_channel(info_bits, EbN0_dB, R=1.0)
            errors_uncoded += np.sum(info_bits != rx_uncoded)
            total_uncoded += n_bits

            # ==============================================================
            #  Canal con codificación convolucional (R = 0.5)
            # ==============================================================
            # 1. Codificar
            coded_bits = conv_encoding(info_bits, gen_matrix)

            # 2. Transmitir por canal AWGN (con tasa R para ajustar ruido)
            _, rx_coded_hard = awgn_channel(coded_bits, EbN0_dB, R=R)

            # 3. Comparar bits codificados
            n_compare = len(rx_coded_hard)
            errors_coded += np.sum(
                coded_bits != rx_coded_hard
            )
            total_coded += n_compare

            # 4. Decodificar con Viterbi
            decoded_bits = viterbi_decode_101_111(rx_coded_hard)

            # 5. Comparar solo los bits de información (descartar cola)
            n_compare = min(n_bits, len(decoded_bits))
            errors_coded_e2e += np.sum(
                info_bits[:n_compare] != decoded_bits[:n_compare]
            )
            total_coded_e2e += n_compare

        ber_uncoded[idx] = errors_uncoded / total_uncoded
        ber_coded[idx] = errors_coded / total_coded
        ber_coded_e2e[idx] = errors_coded_e2e / total_coded_e2e

        print(f"  Eb/N0 = {EbN0_dB:5.1f} dB | "
              f"BER sin cod. = {ber_uncoded[idx]:.2e} | "
              f"BER con cod. = {ber_coded_e2e[idx]:.2e}")

    # ======================================================================
    #  Curvas teóricas
    # ======================================================================
    EbN0_dB_fine = np.linspace(0, 10, 200)
    EbN0_lin_fine = 10.0 ** (EbN0_dB_fine / 10.0)

    # Pe teórica sin codificar: Q(sqrt(2*Eb/N0))
    pe_uncoded_theo = Q(np.sqrt(2.0 * EbN0_lin_fine))

    # Pe teórica del canal con codificación (probabilidad de error de símbolo
    # en el canal BSC): Q(sqrt(2*R*Eb/N0))
    pe_coded_channel_theo = Q(np.sqrt(2.0 * R * EbN0_lin_fine))

    # ======================================================================
    #  Figura BER vs Eb/N0 (teórica vs. simulación)
    # ======================================================================
    plt.figure(figsize=(9, 6))

    # Filtrar BER = 0 para el gráfico semilogarítmico
    mask_unc = ber_uncoded > 0
    mask_cod = ber_coded > 0

    # Simulación
    if np.any(mask_unc):
        plt.semilogy(EbN0_dB_range[mask_unc], ber_uncoded[mask_unc],
                     'bo-', linewidth=1.5, markersize=6,
                     label='Sin codificar (simulación)')
    if np.any(mask_cod):
        plt.semilogy(EbN0_dB_range[mask_cod], ber_coded[mask_cod],
                     'rs-', linewidth=1.5, markersize=6,
                     label='Conv. R=1/2 Viterbi (simulación)')

    # Teórica
    plt.semilogy(EbN0_dB_fine, pe_uncoded_theo, 'b--', linewidth=1,
                 label=r'Teórica sin cod.: $Q(\sqrt{2E_b/N_0})$')
    plt.semilogy(EbN0_dB_fine, pe_coded_channel_theo, 'r--', linewidth=1,
                 label=r'Teórica canal cod.: $Q(\sqrt{2RE_b/N_0})$')

    plt.xlabel(r'$E_b/N_0$ (dB)')
    plt.ylabel('BER')
    plt.title(r'BER vs $E_b/N_0$: sin codificar vs. código convolucional '
              r'$(2,1,3)$')
    plt.legend(loc='lower left')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.ylim([1e-6, 1])
    plt.xlim([0, 10])
    plt.tight_layout()
    plt.savefig('ber_curve.png', dpi=150)
    print(f"\n  Figura guardada: ber_curve.png")

    # ======================================================================
    #  Figura BER vs Eb/N0
    # ======================================================================
    plt.figure(figsize=(9, 6))

    # Filtrar BER = 0 para el gráfico semilogarítmico
    mask_unc = ber_uncoded > 0
    mask_cod_e2e = ber_coded_e2e > 0

    # Simulación
    if np.any(mask_unc):
        plt.semilogy(EbN0_dB_range[mask_unc], ber_uncoded[mask_unc],
                     'bo-', linewidth=1.5, markersize=6,
                     label='Sin codificar (simulación)')
    if np.any(mask_cod_e2e):
        plt.semilogy(EbN0_dB_range[mask_cod_e2e], ber_coded_e2e[mask_cod_e2e],
                     'rs-', linewidth=1.5, markersize=6,
                     label='Conv. R=1/2 Viterbi (simulación)')

    plt.xlabel(r'$E_b/N_0$ (dB)')
    plt.ylabel('BER')
    plt.title(r'BER vs $E_b/N_0$: sin codificar vs. código convolucional '
              r'$(2,1,3)$')
    plt.legend(loc='lower left')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.ylim([1e-6, 1])
    plt.xlim([0, 10])
    plt.tight_layout()
    plt.savefig('ber_curve_e2e.png', dpi=150)
    print(f"\n  Figura guardada: ber_curve_e2e.png")

    # ======================================================================
    #  Sección 6: Decodificación con cancelación de interferencia
    # ======================================================================
    print("\n" + "=" * 60)
    print("  Decodificación del mensaje con cancelación de interferencia")
    print("=" * 60)

    try:
        message, _ = interf_cancel('observations.txt', 'data_rx.txt')
        print(f"\n  Mensaje decodificado:\n")
        print(f"  {message}\n")
    except FileNotFoundError as e:
        print(f"\n  Error: No se encontraron los archivos de datos: {e}")
        print("  Asegúrese de que observations.txt y data_rx.txt están")
        print("  en el directorio de trabajo.")
    except Exception as e:
        print(f"\n  Error durante la cancelación de interferencia: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("  Simulación completada.")
    print("=" * 60)


if __name__ == '__main__':
    main()