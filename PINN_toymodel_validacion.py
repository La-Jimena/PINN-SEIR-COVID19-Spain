# ==============================================================================
# PINN-SEIR: VALIDACIÓN ESTADÍSTICA SOBRE MODELO SINTÉTICO
#
# Descripción:
#   Implementación completa de una Red Neuronal Informada por la Física (PINN)
#   para el modelo epidemiológico SEIR con tasa de transmisión variable β(t),
#   siguiendo el marco de Nouvellet et al. (2021).
#
#   El script realiza dos experimentos de validación independientes sobre datos
#   sintéticos generados por generate_data_seir_BetaFunction.py:
#
#   Experimento 1 — Robustez frente al ruido en los datos (Tabla 2.2 del TFG):
#       Se entrena la PINN sobre 50 conjuntos de datos sintéticos distintos,
#       cada uno con una realización diferente de ruido gaussiano multiplicativo
#       (amplitud 0.05). En cada iteración se carga un archivo .npz distinto,
#       manteniendo el volumen completo de la muestra para aislar el efecto del
#       ruido sobre la identificación de parámetros.
#
#   Experimento 2 — Estabilidad frente a semillas de inicialización (Tabla 2.3):
#       Se entrena la PINN 50 veces sobre el MISMO archivo de datos sintéticos,
#       variando únicamente la semilla de inicialización aleatoria de los pesos
#       de la red neuronal. Permite cuantificar si el algoritmo converge de forma
#       robusta independientemente del estado inicial del optimizador.
#
#   Visualizaciones generadas:
#       · resultados_fisicos_toymodel.eps : Figura 2.2 del TFG (compartimentos
#         SEIR + Rt identificado frente a movilidad µ(t))
#       · aprendizaje_parametros.eps      : Figura 2.3 del TFG (trayectoria de
#         convergencia de φ y R0 durante las épocas de Adam)
#
# Arquitectura del modelo:
#   - Red de estados (Nstates): perceptrón multicapa con 4 capas ocultas de
#     80 neuronas y activación Tanh. Aproxima S(t), E(t) e I(t) de forma continua.
#   - Parámetros físicos libres: α = γ·R0 (transmisión basal) y φ (sensibilidad
#     a la movilidad), identificados por retropropagación junto con los pesos.
#   - β(t) = α · exp(−φ · (1 − µ(t))), formulación de Nouvellet et al. (2021).
#
# Protocolo de optimización en dos etapas (Sección 2.3.3 del TFG):
#   1. Adam (4000 épocas): exploración global con escalado progresivo del peso
#      de las EDO (curriculum de física), de ωode=1 hasta ωode=1000.
#   2. L-BFGS (50 iteraciones): refinamiento local de alta precisión partiendo
#      de los pesos pre-entrenados por Adam.
#
# Uso:
#   Requiere la carpeta "Archivos_sinteticos/" con los 50 archivos .npz
#   generados previamente por generate_data_seir_BetaFunction.py.
#
# Dependencias: numpy, matplotlib, scipy, torch, pandas, glob
#
# Autores:
#   J. J. Sánchez <jj.sanchez@upm.es> — diseño arquitectónico y metodología
#   J. Blanco — implementación, validación y adaptación al TFG
#   Última actualización: 2026-05-06
#
# Aviso de asistencia de IA:
#   Desarrollado con soporte basado en IA para la optimización arquitectónica,
#   implementación de la inferencia de funciones y validación de datos frente
#   a la literatura epidemiológica.
#
# Referencias:
#   · Nouvellet, P., et al. (2021). Reduction in mobility and COVID-19
#     transmission. Nature Communications, 12, 1090.
#     https://doi.org/10.1038/s41467-021-21358-2
#   · Grimm, V., et al. (2022). Estimating the time-dependent contact rate of
#     SIR and SEIR models using PINNs. ETNA, 56, 1-27.
#     https://doi.org/10.1553/etna_vol56s1
#   · Raissi, M., et al. (2019). Physics-informed neural networks. J. Comput.
#     Phys., 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
#   · Millevoi, C., et al. (2024). A PINN approach for compartmental
#     epidemiological models. PLOS Comput. Biol., 20(9), e1012387.
#   · Aràndiga, F., et al. (2020). A Spatial-Temporal Model for the Evolution
#     of the COVID-19 Pandemic in Spain Including Mobility. Mathematics, 8, 1677.
#     https://doi.org/10.3390/math8101677
# ==============================================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch
import torch.nn as nn
import pandas as pd

# ==============================================================================
# SECCIÓN A: CONFIGURACIÓN GLOBAL
# ==============================================================================

# --- Entorno de trabajo ---
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except NameError:
    dname = os.getcwd()

# --- Semilla base de reproducibilidad ---
# Se usa como punto de partida; cada experimento la modifica según corresponda.
random_seed_value = 24
torch.manual_seed(random_seed_value)
np.random.seed(random_seed_value)

# --- Dispositivo de cómputo ---
# Usa GPU si está disponible; en caso contrario, CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Parámetros epidemiológicos fijos (Tabla 2.1 del TFG) ---
# Tomados de Nouvellet et al. (2021) y Grimm et al. (2022).
# Se fijan antes del entrenamiento y no se optimizan junto con α y φ.
GAMMA_REF = 0.15          # Tasa de recuperación (periodo infeccioso medio: 6.48 días)
KAPPA     = 0.2           # Tasa de incubación (periodo latente medio: 5 días)
B_REAL    = 2.5           # Valor teórico de φ para validación
R0_REAL   = 3.0           # Valor teórico de R0 para validación

# ==============================================================================
# SECCIÓN B: CARGA DE DATOS SINTÉTICOS
# ==============================================================================

# Los archivos .npz fueron generados por generate_data_seir_BetaFunction.py.
# Cada archivo contiene una realización independiente del modelo SEIR con ruido
# gaussiano multiplicativo de amplitud 0.05 sobre la solución limpia.
carpeta_datos        = "Archivos_sinteticos"
archivos_disponibles = sorted(glob.glob(os.path.join(carpeta_datos, "data_SEIR_run*.npz")))

if len(archivos_disponibles) == 0:
    raise FileNotFoundError(
        f"No se encontraron archivos en '{carpeta_datos}/data_SEIR_run*.npz'. "
        "Ejecuta primero generate_data_seir_BetaFunction.py para generarlos."
    )

# Se limita a 50 realizaciones, que es el tamaño del experimento de Monte Carlo.
n_ejecuciones = min(50, len(archivos_disponibles))
print(f"✓ {n_ejecuciones} archivos sintéticos encontrados en '{carpeta_datos}/'")

# ==============================================================================
# SECCIÓN C: ARQUITECTURA DE LA PINN
# ==============================================================================

class NeuralNetwork(nn.Module):
    """
    Red de estados (Nstates): aproxima el mapa continuo t → (S, E, I).

    Arquitectura: perceptrón multicapa con una capa de entrada, num_hidden capas
    ocultas de num_neurons neuronas cada una, y una capa de salida con 3 neuronas
    (una por compartimento). La función de activación Tanh garantiza derivadas
    continuas de alto orden, requisito indispensable para el cálculo de residuos
    de las EDO mediante diferenciación automática (Autograd).
    """
    def __init__(self, num_inputs=1, num_outputs=3, num_neurons=80, num_hidden=4):
        super().__init__()
        self.fc_in  = nn.Linear(num_inputs, num_neurons)
        self.hidden = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden)]
        )
        self.act    = nn.Tanh()
        self.fc_out = nn.Linear(num_neurons, num_outputs)

    def forward(self, x):
        """Propagación hacia adelante."""
        out = self.act(self.fc_in(x))
        for layer in self.hidden:
            out = self.act(layer(out))
        return self.fc_out(out)


def mu_pinn(t):
    """
    Función de movilidad sintética µ(t): forzamiento exógeno que modula β(t).

    Simula el patrón observado durante la primera ola del COVID-19 en España:
    caída exponencial durante el confinamiento, seguida de recuperación lineal
    tras el levantamiento de restricciones. La transición entre fases se
    implementa mediante una sigmoide para evitar discontinuidades que
    dificultarían la diferenciación automática de las EDO.

    Parámetros:
        t_cambio     = 55.0  (día del cambio de régimen)
        k_suave      = 0.5   (suavidad de la transición sigmoidal)
        valor_minimo = 0.37  (movilidad mínima durante el confinamiento estricto)
    """
    t_cambio, k_suave, valor_minimo = 55.0, 0.5, 0.37

    # Fase 1: caída exponencial desde µ=1 hasta µ=valor_minimo
    mu_caida = valor_minimo + (1.0 - valor_minimo) * torch.exp(-0.08 * t)

    # Fase 2: recuperación lineal a partir del día t_cambio
    mu0      = valor_minimo + (1.0 - valor_minimo) * torch.exp(torch.tensor(-0.08 * t_cambio))
    mu_recup = mu0 + 0.005 * (t - t_cambio)

    # Transición suave entre fases mediante función sigmoide
    switch = torch.sigmoid(k_suave * (t - t_cambio))
    return torch.clamp((1 - switch) * mu_caida + switch * mu_recup, 0.0, 1.0)


class SEIR_PINN(nn.Module):
    """
    Módulo unificado PINN-SEIR.

    Encapsula la red de estados (NeuralNetwork) y los dos parámetros físicos
    libres a identificar:
        - alpha_raw: forma sin restricciones de α = γ·R0 (transmisión basal)
        - B_raw:     forma sin restricciones de φ (sensibilidad a la movilidad)

    Ambos parámetros se almacenan en espacio no restringido y se transforman
    mediante Softplus antes de usarlos, garantizando positividad sin imponer
    un límite superior artificial. La inicialización en 1.0 es deliberadamente
    neutra: no aporta información previa al optimizador sobre el valor esperado,
    lo que valida la capacidad de convergencia autónoma del algoritmo.
    """
    def __init__(self):
        super().__init__()
        self.net       = NeuralNetwork()
        self.B_raw     = nn.Parameter(torch.tensor([1.0]))   # φ latente
        self.alpha_raw = nn.Parameter(torch.tensor([1.0]))   # α latente

    def get_beta(self, t):
        """
        β(t) = α · exp(−φ · (1 − µ(t))) siguiendo Nouvellet et al. (2021).
        Softplus garantiza α > 0 y φ > 0 sin imponer un límite superior artificial.
        """
        m_t   = mu_pinn(t)
        B     = torch.nn.functional.softplus(self.B_raw)
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        return alpha * torch.exp(-B * (1.0 - m_t))

    def forward(self, x):
        """Normalización temporal: dominio [0, 360] → [0, 1]."""
        return self.net(x / 360.0)


def pinnModel(model, t):
    """
    Transforma las salidas brutas de la red en compartimentos físicos S, E, I.

    Aplica Hard Constraints (ec. 2.6 del TFG) para que las condiciones iniciales
    se satisfagan exactamente por construcción, sin penalizarlas en la función
    de pérdida (Millevoi et al., 2024):

        S(t) = S0 + φ(t) · s_raw(t)
        E(t) = E0 + φ(t) · Softplus(e_raw(t))
        I(t) = I0 + φ(t) · Softplus(i_raw(t))

    donde φ(t) = t/360 se anula en t=0, forzando S(0)=S0, E(0)=E0, I(0)=I0.
    Softplus en E e I garantiza positividad biológica en todo el dominio temporal.

    Las variables globales So, Eo, Io deben estar definidas antes de llamar
    a esta función (se actualizan al inicio de cada iteración del bucle).
    """
    t_norm          = t / 360.0
    out             = model.net(t_norm)
    s_raw, e_raw, i_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # phi = t/360: factor que se anula en t=0 para imponer condiciones iniciales
    phi = t / 360.0
    S   = So + phi * s_raw
    E   = Eo + phi * torch.nn.functional.softplus(e_raw)
    I   = Io + phi * torch.nn.functional.softplus(i_raw)
    return S, E, I


def init_weights(m):
    """
    Inicialización Xavier uniforme para los pesos de la red.
    Previene el desvanecimiento o explosión de gradientes en las primeras épocas.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

# ==============================================================================
# SECCIÓN D: FUNCIONES DE PÉRDIDA
# ==============================================================================

def compute_loss_ode(model, t):
    """
    Término de pérdida de física (Lode): residuos de las EDO SEIR (ec. 2.9-2.12).

    Calcula las derivadas temporales exactas dS/dt, dE/dt, dI/dt mediante
    diferenciación automática (Autograd) y las compara con el lado derecho
    de las ecuaciones del modelo SEIR.

    Incluye un anclaje en t=0 con peso 100 para estabilizar la fase inicial
    del entrenamiento, complementando los Hard Constraints de pinnModel durante
    las primeras épocas cuando los gradientes de phi son muy pequeños.
    """
    loss_fn = nn.MSELoss()
    S, E, I = pinnModel(model, t)
    beta    = model.get_beta(t)

    # Anclaje en t=0: estabiliza las primeras épocas de Adam
    t0 = torch.tensor([[0.0]], device=device, requires_grad=True)
    S0_p, E0_p, I0_p = pinnModel(model, t0)
    loss_init = (loss_fn(S0_p, torch.tensor([[So]], device=device)) +
                 loss_fn(E0_p, torch.tensor([[Eo]], device=device)) +
                 loss_fn(I0_p, torch.tensor([[Io]], device=device)))

    # Derivadas exactas mediante diferenciación automática
    St = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True)[0]
    Et = torch.autograd.grad(E, t, torch.ones_like(E), create_graph=True)[0]
    It = torch.autograd.grad(I, t, torch.ones_like(I), create_graph=True)[0]

    # Residuos de las tres EDO del modelo SEIR
    loss_St = loss_fn(St, -beta * S * I)
    loss_Et = loss_fn(Et,  beta * S * I - κ * E)
    loss_It = loss_fn(It,  κ * E - γ * I)

    return loss_St + loss_Et + loss_It + 100.0 * loss_init


def compute_loss_data(model, data_t, data_S, data_E, data_I):
    """
    Término de pérdida de datos (Ldata): MSE ponderado (ec. 2.8 del TFG).

        Ldata = wS·MSE(S) + wE·MSE(E) + wI·MSE(I)
    con wS=10, wE=10, wI=100.

    El peso mayor en I (100 frente a 10) compensa que el compartimento de
    infectados tiene magnitudes mucho menores que el de susceptibles, forzando
    al optimizador a ajustar el pico de infectados, que es el observable
    epidemiológico más relevante.
    """
    S, E, I = pinnModel(model, data_t)
    return (10.0  * torch.mean((S - data_S)**2) +
            10.0  * torch.mean((E - data_E)**2) +
            100.0 * torch.mean((I - data_I)**2))

# ==============================================================================
# SECCIÓN E: EXPERIMENTO 1 — ROBUSTEZ FRENTE AL RUIDO EN LOS DATOS
# ==============================================================================
#
# Objetivo: cuantificar el error de identificación de α y φ cuando los datos
# de entrenamiento tienen diferentes realizaciones de ruido estocástico.
# En cada iteración se carga un archivo .npz distinto (ruido diferente),
# manteniendo el volumen completo de la muestra para aislar esta fuente de
# incertidumbre. Resultado: Tabla 2.2 del TFG.
# ==============================================================================

print("\n" + "="*70)
print("  EXPERIMENTO 1: ROBUSTEZ FRENTE AL RUIDO EN LOS DATOS (N=50)")
print("="*70)

lista_B_ruido   = []
lista_R0_ruido  = []
num_epochs_Adam = 4000

for i in range(n_ejecuciones):
    fichero_actual = archivos_disponibles[i]
    raw            = np.load(fichero_actual)
    print(f"\n>>> Iteración {i+1}/{n_ejecuciones} | Archivo: {fichero_actual}")

    # La semilla varía con i para reproducibilidad controlada;
    # lo que cambia entre iteraciones es el archivo .npz (diferente ruido).
    torch.manual_seed(random_seed_value + i)
    np.random.seed(random_seed_value + i)

    # Carga de los primeros 200 puntos de los 360 generados (ventana de entrenamiento)
    t_data_np = raw['t'][:200].astype(np.float32)
    S_data_np = raw['S_noisy'][:200].astype(np.float32)
    E_data_np = raw['E_noisy'][:200].astype(np.float32)
    I_data_np = raw['I_noisy'][:200].astype(np.float32)

    # Condiciones iniciales y parámetros fijos del archivo actual
    So, Eo, Io, Ro = float(raw['S0']), float(raw['E0']), float(raw['I0']), float(raw['R0'])
    κ, γ           = float(raw['kappa']), float(raw['gamma'])

    # Conversión a tensores PyTorch
    data_t_iter = torch.tensor(t_data_np, device=device).view(-1, 1).requires_grad_(True)
    data_S_iter = torch.tensor(S_data_np, device=device).view(-1, 1)
    data_E_iter = torch.tensor(E_data_np, device=device).view(-1, 1)
    data_I_iter = torch.tensor(I_data_np, device=device).view(-1, 1)

    # Puntos de colocación para los residuos de las EDO (Latin Hypercube Sampling)
    # Distribuidos uniformemente en [0, 200] para cobertura homogénea del dominio.
    sampler = qmc.LatinHypercube(d=1)
    t_ode   = torch.tensor(
        sampler.random(n=2000) * 200.0,
        dtype=torch.float32, device=device, requires_grad=True
    )

    model = SEIR_PINN().to(device)
    model.net.apply(init_weights)

    # Tasas de aprendizaje diferenciadas: lr más alto para α y φ porque su
    # espacio de búsqueda es más acotado que el de los pesos de la red.
    optimizer = torch.optim.Adam([
        {"params": model.net.parameters(),         "lr": 1e-4},
        {"params": [model.B_raw, model.alpha_raw], "lr": 1e-3}
    ])

    # --- Bucle Adam con curriculum de física ---
    # ωode se incrementa progresivamente de 1 a 1000 para guiar al optimizador
    # desde el ajuste a datos hacia soluciones biológicamente consistentes.
    for epoch in range(num_epochs_Adam):
        if   epoch < 500:  w_ode = 1.0
        elif epoch < 1500: w_ode = 50.0
        elif epoch < 3000: w_ode = 200.0
        else:              w_ode = 1000.0

        optimizer.zero_grad()
        l_ode  = compute_loss_ode(model, t_ode)
        l_data = compute_loss_data(model, data_t_iter,
                           torch.tensor(S_data_np, device=device).view(-1,1),
                           torch.tensor(E_data_np, device=device).view(-1,1),
                           torch.tensor(I_data_np, device=device).view(-1,1))
        # Ltotal = ωode·Lode + 100·Ldata (ec. 2.7 del TFG)
        total_loss = w_ode * l_ode + 100.0 * l_data
        total_loss.backward()
        optimizer.step()

    # --- Refinamiento L-BFGS ---
    # Partiendo de los pesos pre-entrenados por Adam, realiza un ajuste local
    # de alta resolución que perfecciona los decimales finales de α y φ.
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure_ruido():
        optimizer_lbfgs.zero_grad()
        loss = (500.0 * compute_loss_ode(model, t_ode) +
                compute_loss_data(model, data_t_iter, data_S_iter, data_E_iter, data_I_iter))
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure_ruido)

    with torch.no_grad():
        alpha_f = torch.nn.functional.softplus(model.alpha_raw).item()
        B_f     = torch.nn.functional.softplus(model.B_raw).item()
        R0_f    = alpha_f / GAMMA_REF
        lista_B_ruido.append(B_f)
        lista_R0_ruido.append(R0_f)
        print(f"   ✓ φ={B_f:.4f} | R0={R0_f:.4f}")

# --- Reporte estadístico: Experimento 1 (Tabla 2.2) ---
arr_B_r  = np.array(lista_B_ruido)
arr_R0_r = np.array(lista_R0_ruido)

print(f"\n{'='*70}")
print(f"    INFORME DE ROBUSTEZ ESTADÍSTICA — RUIDO (N={n_ejecuciones})")
print(f"{'='*70}")
print(f"  Parámetro φ (sensibilidad a la movilidad):")
print(f"    Media identificada : {np.mean(arr_B_r):.4f}")
print(f"    Varianza           : {np.var(arr_B_r):.4f}")
print(f"    Desviación típica  : ± {np.std(arr_B_r):.4f}")
print(f"    Valor real         : {B_REAL}")
print(f"    Error relativo     : {abs(np.mean(arr_B_r) - B_REAL)/B_REAL * 100:.2f} %")
print(f"{'-'*70}")
print(f"  Parámetro R0 (número reproductivo básico):")
print(f"    Media identificada : {np.mean(arr_R0_r):.4f}")
print(f"    Varianza           : {np.var(arr_R0_r):.4f}")
print(f"    Desviación típica  : ± {np.std(arr_R0_r):.4f}")
print(f"    Valor real         : {R0_REAL}")
print(f"    Error relativo     : {abs(np.mean(arr_R0_r) - R0_REAL)/R0_REAL * 100:.2f} %")
print(f"{'='*70}")

# ==============================================================================
# SECCIÓN F: EXPERIMENTO 2 — ESTABILIDAD FRENTE A SEMILLAS DE INICIALIZACIÓN
# ==============================================================================
#
# Objetivo: comprobar que la convergencia del algoritmo es independiente del
# estado inicial aleatorio de los pesos de la red neuronal.
#
# Se usa SIEMPRE el mismo archivo .npz (archivos_disponibles[0]), de modo que
# el único factor que varía entre iteraciones es la semilla de inicialización
# de los pesos de NeuralNetwork. Los parámetros físicos α y φ se inicializan
# siempre en 1.0 (valor neutro). Resultado: Tabla 2.3 del TFG.
# La trayectoria de convergencia de la iteración 50 genera la Figura 2.3.
# ==============================================================================

print("\n" + "="*70)
print("  EXPERIMENTO 2: ESTABILIDAD FRENTE A SEMILLAS DE INICIALIZACIÓN (N=50)")
print("="*70)

lista_B_semillas  = []
lista_R0_semillas = []
historia_B        = []   # Trayectoria de φ durante Adam (última iteración → Fig. 2.3)
historia_R0       = []   # Trayectoria de R0 durante Adam (última iteración → Fig. 2.3)

# Archivo fijo: siempre el primero de la lista.
# Esto aísla el efecto de la semilla del efecto del ruido (que varía entre archivos).
fichero_fijo = archivos_disponibles[0]
raw          = np.load(fichero_fijo)
print(f"⏺ Archivo fijo para este experimento: {fichero_fijo}")

t_data_np = raw['t'][:200].astype(np.float32)
S_data_np = raw['S_noisy'][:200].astype(np.float32)
E_data_np = raw['E_noisy'][:200].astype(np.float32)
I_data_np = raw['I_noisy'][:200].astype(np.float32)

So, Eo, Io, Ro = float(raw['S0']), float(raw['E0']), float(raw['I0']), float(raw['R0'])
κ, γ           = float(raw['kappa']), float(raw['gamma'])

# Tensores fijos para todo el experimento (los datos no cambian entre iteraciones)
data_t_iter = torch.tensor(t_data_np, device=device).view(-1, 1).requires_grad_(True)
data_S_iter = torch.tensor(S_data_np, device=device).view(-1, 1)
data_E_iter = torch.tensor(E_data_np, device=device).view(-1, 1)
data_I_iter = torch.tensor(I_data_np, device=device).view(-1, 1)

for i in range(50):
    print(f"\n>>> Ejecución {i+1}/50")

    # Semilla diferente en cada vuelta: modifica únicamente la inicialización
    # de los pesos de NeuralNetwork. Los parámetros físicos siguen en 1.0.
    nueva_semilla = random_seed_value + i
    torch.manual_seed(nueva_semilla)
    np.random.seed(nueva_semilla)

    # Puntos de colocación regenerados con la nueva semilla
    sampler = qmc.LatinHypercube(d=1)
    t_ode   = torch.tensor(
        sampler.random(n=2000) * 200.0,
        dtype=torch.float32, device=device, requires_grad=True
    )

    # Nueva instancia del modelo: aplica la nueva semilla a los pesos de la red
    model = SEIR_PINN().to(device)
    model.net.apply(init_weights)

    optimizer = torch.optim.Adam([
        {"params": model.net.parameters(),         "lr": 1e-4},
        {"params": [model.B_raw, model.alpha_raw], "lr": 1e-3}
    ])

    # Bucle Adam con curriculum de física (igual que Experimento 1)
    for epoch in range(num_epochs_Adam):
        if   epoch < 500:  w_ode = 1.0
        elif epoch < 1500: w_ode = 50.0
        elif epoch < 3000: w_ode = 200.0
        else:              w_ode = 1000.0

        optimizer.zero_grad()
        l_ode  = compute_loss_ode(model, t_ode)
        l_data = compute_loss_data(model, data_t_iter, data_S_iter, data_E_iter, data_I_iter)
        total_loss = w_ode * l_ode + 100.0 * l_data
        total_loss.backward()
        optimizer.step()

        # Solo en la iteración 50 (i=49) se registra la trayectoria de convergencia
        # para generar la Figura 2.3 del TFG.
        if i == 49:
            with torch.no_grad():
                B_ev     = torch.nn.functional.softplus(model.B_raw).item()
                alpha_ev = torch.nn.functional.softplus(model.alpha_raw).item()
                historia_B.append(B_ev)
                historia_R0.append(alpha_ev / GAMMA_REF)

    # Refinamiento L-BFGS
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure_semillas():
        optimizer_lbfgs.zero_grad()
        loss = (500.0 * compute_loss_ode(model, t_ode) +
                compute_loss_data(model, data_t_iter, data_S_iter, data_E_iter, data_I_iter))
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure_semillas)

    with torch.no_grad():
        alpha_f = torch.nn.functional.softplus(model.alpha_raw).item()
        B_f     = torch.nn.functional.softplus(model.B_raw).item()
        R0_f    = alpha_f / GAMMA_REF
        lista_B_semillas.append(B_f)
        lista_R0_semillas.append(R0_f)
        print(f"   ✓ Semilla {nueva_semilla} | φ={B_f:.4f} | R0={R0_f:.4f}")

# --- Reporte estadístico: Experimento 2 (Tabla 2.3) ---
arr_B_s  = np.array(lista_B_semillas)
arr_R0_s = np.array(lista_R0_semillas)

print(f"\n{'='*70}")
print(f"    INFORME DE ESTABILIDAD ANTE SEMILLAS (N=50)")
print(f"{'='*70}")
print(f"  Parámetro φ (sensibilidad a la movilidad):")
print(f"    Media identificada : {np.mean(arr_B_s):.4f}")
print(f"    Varianza           : {np.var(arr_B_s):.4f}")
print(f"    Desviación típica  : ± {np.std(arr_B_s):.4f}")
print(f"    Valor real         : {B_REAL}")
print(f"    Error relativo     : {abs(np.mean(arr_B_s) - B_REAL)/B_REAL * 100:.2f} %")
print(f"{'-'*70}")
print(f"  Parámetro R0 (número reproductivo básico):")
print(f"    Media identificada : {np.mean(arr_R0_s):.4f}")
print(f"    Varianza           : {np.var(arr_R0_s):.4f}")
print(f"    Desviación típica  : ± {np.std(arr_R0_s):.4f}")
print(f"    Valor real         : {R0_REAL}")
print(f"    Error relativo     : {abs(np.mean(arr_R0_s) - R0_REAL)/R0_REAL * 100:.2f} %")
print(f"{'='*70}")

# ==============================================================================
# SECCIÓN G: VISUALIZACIÓN — FIGURA 2.2 DEL TFG
# ==============================================================================
#
# Panel de 5 subgráficos (a–d compartimentos SEIR + e Rt vs movilidad) que
# muestra el ajuste de la PINN sobre los datos sintéticos de la última iteración
# del Experimento 2. Los parámetros φ y R0 identificados se muestran en el
# título del panel e).
# ==============================================================================

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def plot_pinn_results_toymodel(model, t_data, S_data, E_data, I_data, R_data):
    """
    Genera el panel de resultados de la PINN sobre el modelo sintético.

    Parámetros
    ----------
    model  : instancia de SEIR_PINN ya entrenada (última iteración MC)
    t_data : tensor (N,1) tiempos de entrenamiento
    S_data…R_data : tensores (N,1) estados con ruido del dataset sintético
    """
    t_max  = float(t_data.max())
    t_eval = torch.linspace(0, t_max, 1000).view(-1, 1).to(device)

    with torch.no_grad():
        S_pred, E_pred, I_pred = pinnModel(model, t_eval)
        R_pred    = 1.0 - (S_pred + E_pred + I_pred)
        B_val     = torch.nn.functional.softplus(model.B_raw).item()
        alpha_val = torch.nn.functional.softplus(model.alpha_raw).item()
        R0_val    = alpha_val / GAMMA_REF
        beta_t    = model.get_beta(t_eval)
        Rt_pred   = beta_t / GAMMA_REF

    def to_np(x):
        return x.detach().cpu().numpy().flatten()

    # --- Layout: 2×2 (a–d) + fila inferior completa (e) ---
    fig = plt.figure(figsize=(18, 15))
    gs  = fig.add_gridspec(3, 2)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_rt = fig.add_subplot(gs[2, :])

    titles  = ['a) S (Susceptibles)', 'b) E (Expuestos)',
               'c) I (Infectados)',   'd) R (Recuperados)']
    data_y  = [to_np(S_data), to_np(E_data), to_np(I_data), to_np(R_data)]
    pred_y  = [to_np(S_pred), to_np(E_pred), to_np(I_pred), to_np(R_pred)]
    colors  = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

    # --- Paneles a–d: compartimentos SEIR ---
    for j, ax in enumerate(axes):
        ax.scatter(to_np(t_data), data_y[j],
                   color=colors[j], s=20, alpha=0.3, label='Datos con Ruido')
        ax.plot(to_np(t_eval), pred_y[j],
                color=colors[j], lw=3, label='Ajuste PINN')
        ax.set_title(titles[j], fontsize=20, fontweight='bold', loc='left')
        ax.set_xlabel('Tiempo (días)', fontsize=18)
        ax.set_ylabel('Población normalizada', fontsize=16)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

    # --- Panel e: Rt identificado vs movilidad µ(t) ---
    ax_rt_sec = ax_rt.twinx()
    lns1 = ax_rt.plot(to_np(t_eval), to_np(Rt_pred),
                      color='purple', lw=4,
                      label=f'$R_t$ Identificado ($R_0$={R0_val:.2f})')
    ax_rt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5,
                  label='Umbral Crítico ($R_t=1$)')
    lns2 = ax_rt_sec.plot(to_np(t_eval), to_np(mu_pinn(t_eval)),
                          color='gray', lw=2, linestyle=':',
                          label='Movilidad $\\mu(t)$')

    ax_rt.set_title(
        f'e) Dinámica de Transmisión Identificada Modelo Sintético '
        f'($\\phi$={B_val:.3f}, $R_0$={R0_val:.2f})',
        fontsize=20, fontweight='bold', loc='left'
    )
    ax_rt.set_xlabel('Tiempo (días)', fontsize=18)
    ax_rt.set_ylabel('Número Reproductivo $R_t$', fontsize=18)
    ax_rt_sec.set_ylabel('Índice de Movilidad $\\mu$', fontsize=18, color='gray')

    lns  = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax_rt.legend(lns, labs, loc='upper right', fontsize=14)
    ax_rt.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig("resultados_fisicos_toymodel.eps", format="eps", bbox_inches='tight')
    plt.show()
    print("✓ Figura guardada: resultados_fisicos_toymodel.eps")


# Ejecución con los datos de la última iteración del Experimento 2
with torch.no_grad():
    data_R_iter = 1.0 - (data_S_iter + data_E_iter + data_I_iter)

print("\n⏺ Generando Figura 2.2: panel de resultados PINN — modelo sintético...")
plot_pinn_results_toymodel(
    model, data_t_iter, data_S_iter, data_E_iter, data_I_iter, data_R_iter
)

# ==============================================================================
# SECCIÓN H: VISUALIZACIÓN — FIGURA 2.3 DEL TFG
# ==============================================================================
#
# Trayectoria de convergencia de φ y R0 durante las épocas de Adam para la
# iteración 50 del Experimento 2, partiendo de inicialización neutra 1.0.
# Demuestra que el algoritmo converge hacia los valores reales desde un estado
# inicial sin información previa sobre los parámetros esperados.
# ==============================================================================

print("\n⏺ Generando Figura 2.3: evolución temporal del aprendizaje...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Panel a): evolución de φ
ax1.plot(historia_B, color='darkblue', lw=2, label=r'$\phi$ Identificado')
ax1.axhline(y=B_REAL, color='red', linestyle='--', lw=2,
            label=f'$\\phi$ Real ({B_REAL})')
ax1.set_title(r"a) Evolución del parámetro $\phi$",
              fontsize=18, fontweight='bold', loc='left')
ax1.set_xlabel("Épocas de optimización (Adam)", fontsize=16)
ax1.set_ylabel("Magnitud del parámetro", fontsize=16)
ax1.set_xlim(0, num_epochs_Adam)
ax1.legend(fontsize=14)
ax1.grid(True, alpha=0.3)

# Panel b): evolución de R0
ax2.plot(historia_R0, color='darkgreen', lw=2, label='$R_0$ Identificado')
ax2.axhline(y=R0_REAL, color='red', linestyle='--', lw=2,
            label=f'$R_0$ Real ({R0_REAL})')
ax2.set_title(r"b) Evolución del parámetro $R_0$",
              fontsize=18, fontweight='bold', loc='left')
ax2.set_xlabel("Épocas de optimización (Adam)", fontsize=16)
ax2.set_ylabel("Magnitud del parámetro", fontsize=16)
ax2.set_xlim(0, num_epochs_Adam)
ax2.legend(fontsize=14)
ax2.grid(True, alpha=0.3)

plt.subplots_adjust(wspace=0.3)
plt.savefig("aprendizaje_parametros.eps", format="eps", bbox_inches='tight')
plt.show()
print("✓ Figura guardada: aprendizaje_parametros.eps")
