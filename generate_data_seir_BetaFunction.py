# ==============================================================================
# GENERACIÓN DE DATOS SINTÉTICOS SEIR CON TASA DE TRANSMISIÓN VARIABLE β(t)
#
# Descripción:
#   Este script genera los 50 conjuntos de datos sintéticos necesarios para
#   los experimentos de validación estadística de la PINN (Apéndice A).
#   Cada conjunto se obtiene integrando numéricamente el sistema de EDO del
#   modelo SEIR con una tasa de transmisión variable β(t) y añadiendo ruido
#   gaussiano multiplicativo de amplitud 0.05 sobre la solución limpia.
#
#   Los archivos generados sirven para dos experimentos independientes:
#
#   Experimento 1 — Robustez frente al ruido (Tabla 2.2 del TFG):
#       La PINN se entrena 50 veces, cada vez con un archivo .npz distinto.
#       Cada archivo contiene una realización diferente del ruido estocástico,
#       manteniendo la solución limpia subyacente constante. Esto permite
#       cuantificar el error de identificación de parámetros atribuible
#       exclusivamente al ruido en los datos de entrenamiento.
#
#   Experimento 2 — Estabilidad frente a semillas de inicialización (Tabla 2.3):
#       La PINN se entrena 50 veces usando siempre el mismo archivo
#       (archivos_disponibles[0] en PINN_toymodel_validacion.py), variando
#       únicamente la semilla de inicialización de los pesos de la red neuronal.
#       Para este experimento, los 50 archivos generados no son estrictamente
#       necesarios (bastaría con uno), pero se generan de forma conjunta para
#       mantener una única ejecución de este script.
#
# Modelo epidemiológico:
#   Se integra el sistema SEIR estándar (ec. 2.1 del TFG):
#       dS/dt = −β(t)·S·I
#       dE/dt =  β(t)·S·I − κ·E
#       dI/dt =  κ·E − γ·I
#       dR/dt =  γ·I
#
#   La tasa de transmisión sigue el marco de Nouvellet et al. (2021)
#   (ec. 2.5 del TFG):
#       β(t) = α · exp(−φ · (1 − µ(t)))
#   donde α = γ·R0 es la transmisión basal y µ(t) es el índice de movilidad.
#
# Parámetros del modelo (Tabla 2.1 del TFG):
#   γ = 0.15   (tasa de recuperación; periodo infeccioso medio: 1/γ ≈ 6.5 días)
#   κ = 0.2    (tasa de incubación; periodo latente medio: 1/κ = 5 días)
#   R0 = 3.0   (número reproductivo básico)
#   φ = 2.5    (sensibilidad de la transmisión frente a la movilidad)
#
# Ruido:
#   Se añade ruido gaussiano multiplicativo de amplitud 0.05 sobre la solución
#   limpia de cada compartimento. Las semillas se inicializan con la hora del
#   sistema operativo para garantizar variabilidad absoluta entre ejecuciones
#   (véase Sección 2.4.1 del TFG).
#
# Salidas:
#   50 archivos .npz en la carpeta "Archivos_sinteticos/", con nomenclatura:
#       data_SEIR_run{i}_seed_{semilla}.npz
#   Cada archivo contiene:
#       t, S_clean, E_clean, I_clean, R_clean  (solución limpia)
#       S_noisy, E_noisy, I_noisy, R_noisy      (solución con ruido)
#       kappa, gamma, S0, E0, I0, R0            (parámetros y condiciones iniciales)
#       noise_amplitude                          (amplitud del ruido aplicado)
#
# Uso:
#   Ejecutar antes de PINN_toymodel_validacion.py. Los archivos generados
#   deben estar en "Archivos_sinteticos/" relativo al directorio de trabajo.
#
# Dependencias: numpy, scipy, matplotlib, os, time, random
#
# Autores:
#   J. J. Sánchez <jj.sanchez@upm.es> — diseño metodológico
#   J. Blanco — implementación y adaptación al TFG
#   Última actualización: 2026-05-06
#
# Aviso de asistencia de IA:
#   Desarrollado con soporte basado en IA para tareas específicas de
#   implementación y validación frente a la literatura epidemiológica.
#
# Referencias:
#   · Nouvellet, P., et al. (2021). Reduction in mobility and COVID-19
#     transmission. Nature Communications, 12, 1090.
#     https://doi.org/10.1038/s41467-021-21358-2
#   · Grimm, V., et al. (2022). Estimating the time-dependent contact rate of
#     SIR and SEIR models using PINNs. ETNA, 56, 1–27.
#     https://doi.org/10.1553/etna_vol56s1
# ==============================================================================

import os
import time
import random
import numpy as np
from scipy.integrate import odeint

# ==============================================================================
# SECCIÓN A: FUNCIONES DEL MODELO
# ==============================================================================

def mu(t):
    """
    Índice de movilidad sintético µ(t): forzamiento exógeno que modula β(t).

    Simula el patrón observado durante la primera ola del COVID-19 en España:
    caída exponencial durante el confinamiento, seguida de recuperación lineal
    tras el levantamiento de restricciones. La transición entre fases se
    implementa mediante una sigmoide para evitar discontinuidades.

    Parámetros internos:
        t_cambio    = 55.0   (día del cambio de régimen)
        k_suave     = 0.5    (suavidad de la transición sigmoidal)
        valor_minimo = 0.37  (movilidad mínima durante el confinamiento estricto)
    """
    t_cambio     = 55.0
    k_suave      = 0.5
    valor_minimo = 0.37

    # Fase 1: caída exponencial desde µ=1 hasta µ=valor_minimo
    mu_caida = valor_minimo + (1.0 - valor_minimo) * np.exp(-0.08 * t)

    # Fase 2: recuperación lineal a partir del día t_cambio
    mu0      = valor_minimo + (1.0 - valor_minimo) * np.exp(-0.08 * t_cambio)
    mu_recup = mu0 + 0.005 * (t - t_cambio)

    # Transición suave entre fases mediante función sigmoide
    switch = 1.0 / (1.0 + np.exp(-k_suave * (t - t_cambio)))

    return np.clip((1.0 - switch) * mu_caida + switch * mu_recup, 0.0, 1.0)


def beta_Nouvellet21(t):
    """
    Tasa de transmisión variable β(t) según Nouvellet et al. (2021).

    Implementa la ecuación 2.5 del TFG:
        β(t) = α · exp(−φ · (1 − µ(t)))
    donde α = γ · R0 agrupa la transmisión basal del patógeno.

    Los parámetros utilizados son los valores de referencia de la Tabla 2.1:
        γ  = 0.15  (tasa de recuperación)
        R0 = 3.0   (número reproductivo básico)
        φ  = 2.5   (sensibilidad a la movilidad)
    """
    gamma = 0.15
    R0    = 3.0
    phi   = 2.5

    alpha = gamma * R0
    return alpha * np.exp(-phi * (1.0 - mu(t)))


def seirModel(y, t, funBeta, kappa, gamma):
    """
    Sistema de EDO del modelo SEIR (ec. 2.1 del TFG).

    Parámetros:
        y      : vector de estado [S, E, I, R]
        t      : instante temporal
        funBeta: función β(t) que devuelve la tasa de transmisión en t
        kappa  : tasa de incubación (κ)
        gamma  : tasa de recuperación (γ)
    """
    S, E, I, R = y
    beta  = funBeta(t)
    dSdt  = -beta * S * I
    dEdt  =  beta * S * I - kappa * E
    dIdt  =  kappa * E - gamma * I
    dRdt  =  gamma * I
    return dSdt, dEdt, dIdt, dRdt

# ==============================================================================
# SECCIÓN B: PARÁMETROS Y CONDICIONES INICIALES
# ==============================================================================

# --- Parámetros epidemiológicos fijos (Tabla 2.1 del TFG) ---
KAPPA          = 0.2    # Tasa de incubación (periodo latente medio: 5 días)
GAMMA          = 0.15   # Tasa de recuperación (periodo infeccioso medio: ~6.5 días)
NOISE_AMPLITUDE = 0.05  # Amplitud del ruido gaussiano multiplicativo

# --- Condiciones iniciales ---
# Al inicio de la simulación se asume que la gran mayoría de la población
# es susceptible y una pequeña fracción se encuentra ya expuesta e infectada.
S0 = 0.99
E0 = 0.005
I0 = 0.005
R0_init = 0.0   # Se renombra para evitar colisión con el parámetro R0 del modelo

# --- Dominio temporal ---
# Se simulan 360 días para disponer de una ventana larga de entrenamiento.
# La PINN usa únicamente los primeros 200 puntos (véase Sección 2.4 del TFG).
N_POINTS = 360
t_array  = np.linspace(0, N_POINTS, N_POINTS)

# --- Carpeta de salida ---
CARPETA_SALIDA = "Archivos_sinteticos"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ==============================================================================
# SECCIÓN C: GENERACIÓN DE LOS 50 ARCHIVOS SINTÉTICOS
# ==============================================================================
#
# En cada iteración se genera una realización independiente del ruido gaussiano
# mediante una semilla derivada de la hora del sistema operativo (en milisegundos).
# Esta estrategia garantiza variabilidad absoluta entre ejecuciones del script,
# de modo que dos ejecuciones consecutivas producen conjuntos de datos distintos
# (véase Sección 2.4.3.1 del TFG). La semilla empleada se incluye en el nombre
# del archivo para facilitar la trazabilidad de cada realización.
#
# Nota: la solución limpia de las EDO es idéntica en los 50 archivos (depende
# únicamente de los parámetros del modelo, que son constantes). Lo que varía
# entre archivos es exclusivamente la realización del ruido añadido.
# ==============================================================================

print("="*65)
print("  GENERACIÓN DE DATOS SINTÉTICOS SEIR (50 realizaciones)")
print("="*65)

for i in range(50):

    # Semilla única basada en la hora del sistema (milisegundos) + índice.
    # La suma de i garantiza que dos iteraciones del mismo bucle no coincidan
    # aunque se ejecuten en el mismo milisegundo.
    seed_value = int(time.time() * 1000) + i
    random.seed(seed_value)

    # --- Integración numérica del sistema SEIR (solución limpia) ---
    sol = odeint(
        seirModel,
        [S0, E0, I0, R0_init],
        t_array,
        args=(beta_Nouvellet21, KAPPA, GAMMA)
    )
    S_clean = sol[:, 0]
    E_clean = sol[:, 1]
    I_clean = sol[:, 2]
    R_clean = sol[:, 3]

    # --- Adición de ruido gaussiano multiplicativo ---
    # El ruido se multiplica por la solución limpia para que su magnitud
    # sea proporcional al valor del compartimento en cada instante. Esto
    # evita que el ruido produzca valores negativos en las fases de baja
    # incidencia, donde los compartimentos E e I son muy pequeños.
    rng   = np.random.default_rng(seed_value)
    noise = rng.normal(loc=0.0, scale=NOISE_AMPLITUDE, size=sol.shape)
    sol_noisy = np.clip(sol * (1.0 + noise), 0.0, 1.0)

    S_noisy = sol_noisy[:, 0]
    E_noisy = sol_noisy[:, 1]
    I_noisy = sol_noisy[:, 2]
    R_noisy = sol_noisy[:, 3]

    # --- Guardado del archivo .npz ---
    nombre_archivo = f"data_SEIR_run{i}_seed_{seed_value}.npz"
    ruta_salida    = os.path.join(CARPETA_SALIDA, nombre_archivo)

    np.savez(
        ruta_salida,
        t          = t_array,
        S_clean    = S_clean,
        E_clean    = E_clean,
        I_clean    = I_clean,
        R_clean    = R_clean,
        S_noisy    = S_noisy,
        E_noisy    = E_noisy,
        I_noisy    = I_noisy,
        R_noisy    = R_noisy,
        kappa      = KAPPA,
        gamma      = GAMMA,
        S0         = S0,
        E0         = E0,
        I0         = I0,
        R0         = R0_init,
        noise_amplitude = NOISE_AMPLITUDE
    )

    print(f"  [{i+1:2d}/50]  Semilla: {seed_value}  →  {nombre_archivo}")

    # Pausa mínima para asegurar que el timestamp cambie entre iteraciones
    time.sleep(0.01)

print("="*65)
print(f"  ✓ 50 archivos guardados en '{CARPETA_SALIDA}/'")
print("="*65)
