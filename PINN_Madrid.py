# ==============================================================================
# PINN-SEIR: INFERENCIA DE LA DINÁMICA DE TRANSMISIÓN DEL COVID-19 EN MADRID
#
# Descripción:
#   Implementación completa de una Red Neuronal Informada por la Física (PINN)
#   para la inferencia de la función de transmisión β(t) y el número reproductivo
#   efectivo Rt durante la primera ola del COVID-19 en la Comunidad de Madrid
#   (2020-03-01 a 2020-06-30), siguiendo el marco de Nouvellet et al. (2021).
#
#   El script sigue el pipeline completo en cinco etapas:
#
#   Sección A — Configuración global:
#       Parámetros epidemiológicos, dispositivo de cómputo y rutas de datos.
#
#   Sección B — Construcción de estados SEIR desde la incidencia real:
#       Los únicos datos observables son la incidencia diaria suavizada y el
#       índice de movilidad de Google. A partir de ellos se reconstruyen los
#       cuatro compartimentos del modelo SEIR mediante tres transformaciones
#       analíticas (C1–C3) que garantizan la consistencia con las EDO del sistema.
#
#   Sección C — Arquitectura de la PINN:
#       Igual que en el modelo sintético, con una diferencia clave: la movilidad
#       µ(t) no es una función analítica sino una señal discreta real (Google
#       Mobility Reports). Se implementa una interpolación lineal diferenciable
#       para que el optimizador pueda calcular gradientes a través de µ(t).
#       El Hard Constraint usa phi = t/T_max, igual que en el modelo sintético,
#       garantizando que S(0)=S0, E(0)=E0, I(0)=I0 por construcción.
#
#   Sección D — Funciones de pérdida adaptadas a datos reales:
#       La función de pérdida de datos se normaliza por el rango de cada
#       compartimento (∆X) para compensar las diferencias de magnitud entre S
#       (~0.01 de variación) e I (~0.002 de variación). Además, se aplica una
#       máscara de pico (días 8–35) con peso ×5 para forzar el ajuste del pico
#       epidémico, que de otro modo quedaría aplastado por la larga cola post-pico.
#       La función de pérdida de física incorpora además un término de conservación
#       de masa d(S+E+I)/dt = −γI y un anclaje en t=0 con peso 1000.
#
#   Sección E — Experimento de Monte Carlo (50 runs, dropout 5%):
#       En cada iteración se descarta aleatoriamente el 5% de los días de datos
#       y se varía la semilla de inicialización, generando la banda de
#       incertidumbre de β(t) que aparece en las Figuras 3.2 y 3.3 del TFG.
#
#   Sección F — Cálculo de Rt y reporte epidemiológico:
#       Interpola β(t) de las 50 iteraciones a resolución diaria y calcula
#       Rt = β(t)/γ. Genera el reporte mensual de la Tabla 3.1 del TFG.
#
#   Sección G — Visualización (Figuras 3.2 del TFG):
#       Panel de 5 subgráficos (a–d compartimentos + e β(t)) con fechas reales,
#       eventos epidemiológicos clave y banda de incertidumbre Monte Carlo.
#
# Diferencias respecto a la PINN del modelo sintético:
#   · La movilidad µ(t) es una señal discreta real interpolada, no una función
#     analítica. Esto requiere el método get_movilidad_continua en SEIR_PINN.
#   · El Hard Constraint usa phi = sqrt(t/T_max) en lugar de t/T_max.
#   · La pérdida de datos no incluye el término de E, ya que E se construye
#     como E = incidencia/κ (derivada de los datos) y añadirla degrada el ajuste.
#   · Se añade un término de conservación de masa en la pérdida de física.
#   · El protocolo de validación usa dropout del 5% en lugar de archivos distintos.
#
# Datos de entrada requeridos:
#   · dataset_SEIR_M_2020-03-01.csv: incidencia diaria suavizada y movilidad
#     de Google para la Comunidad de Madrid. Generado por Unificacion.py.
#     Columnas: ['t', 'suavizado_mean', 'mobility_smooth']
#
# Salidas generadas:
#   · metricas_rt_madrid_v3.csv:         β(t) media ± σ por día (50 runs)
#   · metricas_rt_madrid_montecarlo.csv: Rt medio ± σ por día (50 runs)
#   · resultados_pinn_madrid.eps/.pdf:   Figura 3.2 del TFG
#
# Dependencias: numpy, matplotlib, scipy, torch, pandas
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
#   · Instituto de Salud Carlos III (2020). RENAVE — Informes COVID-19.
#     https://cnecovid.isciii.es/
#   · Google (2020). Community Mobility Reports.
#     https://www.google.com/covid19/mobility/
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import qmc
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import pandas as pd

# ==============================================================================
# SECCIÓN A: CONFIGURACIÓN GLOBAL
# ==============================================================================

# --- Entorno de trabajo ---
carpeta_actual = (os.path.dirname(os.path.abspath(__file__))
                  if '__file__' in dir() else os.getcwd())
fichero = os.path.join(carpeta_actual, 'dataset_SEIR_M_2020-03-01.csv')

# --- Parámetros de población y temporales ---
N_MADRID = 6.66e6    # Población de la Comunidad de Madrid (2020)
T_MAX    = 90.0      # Ventana de análisis: 90 días desde el 1 de marzo de 2020

# --- Dispositivo de cómputo ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Semilla base de reproducibilidad ---
random_seed_value = 24
torch.manual_seed(random_seed_value)
np.random.seed(random_seed_value)

# --- Parámetros epidemiológicos fijos (Nouvellet et al., 2021) ---
# Se fijan antes del entrenamiento y no se optimizan junto con α y φ.
κ = 0.2           # Tasa de incubación: periodo latente medio = 1/κ = 5 días
γ = 1.0 / 6.48   # Tasa de recuperación: periodo infeccioso medio = 6.48 días

# Fecha de inicio para el eje temporal de las figuras
FECHA_INICIO = datetime(2020, 3, 1)

# Eventos epidemiológicos clave (días desde t=0) para las figuras
EVENTOS = {
    't=13 — Estado de alarma (14 mar)':  13,
    't=43 — Fin confinamiento estricto': 43,
    't=61 — Inicio desescalada (1 may)': 61,
}

# ==============================================================================
# SECCIÓN B: CONSTRUCCIÓN DE ESTADOS SEIR DESDE LA INCIDENCIA REAL
# ==============================================================================
#
# En datos reales, los compartimentos S, E, I, R no son directamente observables.
# Solo se dispone de la incidencia diaria de casos confirmados (suavizada) y del
# índice de movilidad de Google. A partir de la incidencia se reconstruyen los
# cuatro compartimentos mediante las transformaciones C1–C3 descritas a continuación,
# consistentes con las EDO del modelo SEIR (Sección 3.2.1 del TFG).
# ==============================================================================

df = pd.read_csv(fichero)
df = df[df['t'] <= T_MAX].copy()
print(f"✓ Dataset: {len(df)} días (Madrid 2020-03-01 → 2020-06-30)")

# Incidencia diaria normalizada por la población total: id(t) = casos(t) / N
# Representa el flujo diario de nuevos infectados como fracción de la población.
incidencia_np = df['suavizado_mean'].values / N_MADRID

n = len(incidencia_np)

# --- C1: Infectados activos I(t) por convolución exponencial (ec. 3.2 del TFG) ---
# En el modelo SEIR, la probabilidad de que alguien infectado el día s siga siendo
# infeccioso el día t es exp(−γ(t−s)). I(t) es la suma ponderada de todos los casos
# pasados por su fracción aún no recuperada. Es más preciso que I = incidencia/γ,
# que asume equilibrio instantáneo y produce señales E e I proporcionales
# (indistinguibles por el optimizador).
I_conv = np.zeros(n, dtype=np.float64)
for t in range(n):
    for s in range(t + 1):
        I_conv[t] += incidencia_np[s] * np.exp(-γ * (t - s))
I_base_np = I_conv.astype(np.float32)

# --- C2: Expuestos E(t) = incidencia / κ (ec. 3.3 del TFG) ---
# De la EDO dI/dt = κE − γI se deduce que, en el régimen donde la incidencia
# observable aproxima el flujo κE, E = incidencia/κ. La relación E/I ya no es
# constante porque I proviene de la convolución (no es proporcional a E).
E_base_np = (incidencia_np / κ).astype(np.float32)

# --- C3: Recuperados R(t) = γ · cumsum(I) (ec. 3.4 del TFG) ---
# Aproximación de la integral de infectados activos mediante suma de Riemann.
R_base_np = (γ * np.cumsum(I_base_np)).astype(np.float32)

# --- Susceptibles S(t) por conservación de masa (ec. 3.5 del TFG) ---
# S = 1 − E − I − R garantiza que S + E + I + R = 1 en todo momento.
S_base_np = (1.0 - E_base_np - I_base_np - R_base_np).astype(np.float32)
S_base_np = np.clip(S_base_np, 0.0, 1.0)   # Evita valores negativos por redondeo

# --- Movilidad suavizada µ(t) (Google Community Mobility Reports) ---
# Promedio de 'workplaces' y 'transit stations', normalizado a [0,1].
# Un valor µ=0.4 indica que la actividad social se redujo al 40% de la normalidad.
m_base_np = df['mobility_smooth'].values.astype(np.float32)
t_base_np = df['t'].values.astype(np.float32)

# --- Condiciones iniciales extraídas de los datos en t=0 ---
Io   = float(I_base_np[0])
Eo   = float(E_base_np[0])
Ro_0 = float(R_base_np[0])
So   = 1.0 - Io - Eo - Ro_0   # S0 ≈ 1 porque la inmunidad inicial es despreciable

print(f"\n  Condiciones iniciales (t=0):")
print(f"    So={So:.6f}  Eo={Eo:.2e}  Io={Io:.2e}  Ro={Ro_0:.2e}")

# ==============================================================================
# SECCIÓN C: ARQUITECTURA DE LA PINN
# ==============================================================================

class NeuralNetwork(nn.Module):
    """
    Red de estados (Nstates): aproxima el mapa continuo t → (S, E, I).

    Arquitectura: perceptrón multicapa con una capa de entrada, 4 capas ocultas
    de 80 neuronas y una capa de salida con 3 neuronas (S, E, I). La función de
    activación Tanh garantiza derivadas continuas de alto orden, requisito
    indispensable para el cálculo de residuos de las EDO mediante Autograd.
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


class SEIR_PINN(nn.Module):
    """
    Módulo unificado PINN-SEIR para datos reales de movilidad.

    A diferencia del modelo sintético, µ(t) no es una función analítica sino
    una señal discreta procedente de los Google Mobility Reports. Para que el
    optimizador pueda calcular gradientes a través de µ(t), se implementa una
    interpolación lineal diferenciable en get_movilidad_continua.

    Los buffers t_base_tensor y m_base_tensor almacenan la señal de movilidad
    registrada en la GPU (si está disponible), evitando transferencias repetidas
    en cada paso de optimización.

    Parámetros físicos libres (inicializados en 1.0, valor neutro):
        alpha_raw → α = γ·R0 (transmisión basal) mediante Softplus
        B_raw     → φ (sensibilidad a la movilidad) mediante Softplus
    """
    def __init__(self, t_base, m_base):
        super().__init__()
        self.net       = NeuralNetwork()
        self.B_raw     = nn.Parameter(torch.tensor([1.0]))
        self.alpha_raw = nn.Parameter(torch.tensor([1.0]))
        # Señal de movilidad registrada como buffer (no optimizable, pero en GPU)
        self.register_buffer('t_base_tensor',
                             torch.tensor(t_base, dtype=torch.float32).view(-1))
        self.register_buffer('m_base_tensor',
                             torch.tensor(m_base, dtype=torch.float32).view(-1))

    def get_movilidad_continua(self, t):
        """
        Interpolación lineal diferenciable de la señal discreta de movilidad µ(t).

        Permite que Autograd calcule gradientes a través de µ(t) durante la
        diferenciación automática de las EDO, lo que sería imposible si se
        usara indexación directa (no diferenciable).
        """
        t_flat   = t.view(-1).clamp(0.0, T_MAX)
        idx_low  = t_flat.floor().long().clamp(0, len(self.m_base_tensor) - 2)
        idx_high = (idx_low + 1).clamp(0, len(self.m_base_tensor) - 1)
        frac     = (t_flat - idx_low.float()).view(-1, 1)
        m_low    = self.m_base_tensor[idx_low].view(-1, 1)
        m_high   = self.m_base_tensor[idx_high].view(-1, 1)
        return m_low + frac * (m_high - m_low)   # interpolación lineal

    def get_beta(self, t):
        """
        β(t) = α · exp(−φ · (1 − µ(t))) siguiendo Nouvellet et al. (2021).
        Softplus garantiza α > 0 y φ > 0 sin imponer un límite superior artificial.
        """
        m_t   = self.get_movilidad_continua(t)
        B     = torch.nn.functional.softplus(self.B_raw)
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        return alpha * torch.exp(-B * (1.0 - m_t))

    def forward(self, x):
        """Normalización temporal: t ∈ [0, T_MAX] → [0, 1]."""
        return self.net(x / T_MAX)


def pinnModel(model, t):
    """
    Transforma las salidas brutas de la red en compartimentos físicos S, E, I.

    Aplica Hard Constraints para que las condiciones iniciales se satisfagan
    exactamente por construcción (Millevoi et al., 2024):

        S(t) = S0 + φ(t) · s_raw(t)
        E(t) = E0 + φ(t) · Softplus(e_raw(t))
        I(t) = I0 + φ(t) · Softplus(i_raw(t))

    phi = t/T_MAX: factor que se anula en t=0, imponiendo las condiciones
    iniciales por construcción sin necesidad de penalizarlas en la función
    de pérdida. Formulación idéntica a la ecuación 2.6 del TFG (t̃ = t/T_max).

    Softplus en E e I garantiza positividad biológica en todo el dominio.
    """
    out             = model(t)
    s_raw, e_raw, i_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # phi = t/T_MAX: factor que se anula en t=0 (Hard Constraint)
    phi = t / T_MAX
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
# SECCIÓN D: FUNCIONES DE PÉRDIDA ADAPTADAS A DATOS REALES
# ==============================================================================

def compute_loss_ode(model, t):
    """
    Término de pérdida de física (Lode): residuos de las EDO SEIR.

    Además de los tres residuos estándar (dS/dt, dE/dt, dI/dt), incluye:

    · Conservación de masa: d(S+E+I)/dt = −γI (equivalente a dR/dt = γI).
      Peso 100. Penaliza violaciones de la restricción S+E+I+R=1 de forma
      diferenciable, sin necesidad de calcular R explícitamente.

    · Anclaje en t=0: penaliza desviaciones de las condiciones iniciales
      con peso 1000 (más alto que en el modelo sintético, donde era 100).
      Necesario porque con datos reales el optimizador tiende a desanclar
      las condiciones iniciales durante las primeras épocas de Adam.
    """
    loss_fn = nn.MSELoss()
    S, E, I = pinnModel(model, t)
    beta    = model.get_beta(t)

    # Derivadas exactas mediante diferenciación automática
    St = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True)[0]
    Et = torch.autograd.grad(E, t, torch.ones_like(E), create_graph=True)[0]
    It = torch.autograd.grad(I, t, torch.ones_like(I), create_graph=True)[0]

    # Residuos de las tres EDO del modelo SEIR
    l_S = loss_fn(St, -beta * S * I)
    l_E = loss_fn(Et,  beta * S * I - κ * E)
    l_I = loss_fn(It,  κ * E - γ * I)

    # Conservación de masa: d(S+E+I)/dt + γI = 0 (equivale a dR/dt = γI)
    # relu evita penalizar pequeñas violaciones negativas (numéricamente inofensivas)
    l_conserv = torch.mean(torch.relu(St + Et + It + γ * I)**2) * 100.0

    # Anclaje fuerte en t=0: complementa los Hard Constraints durante las
    # primeras épocas cuando phi≈0 y los gradientes son muy pequeños.
    t0 = torch.zeros(1, 1, device=device, requires_grad=False)
    S0p, E0p, I0p = pinnModel(model, t0)
    l_ci = (loss_fn(S0p, torch.tensor([[So]], device=device)) +
            loss_fn(E0p, torch.tensor([[Eo]], device=device)) +
            loss_fn(I0p, torch.tensor([[Io]], device=device))) * 1000.0

    return l_S + l_E + l_I + l_conserv + l_ci


def compute_loss_data(model, data_t, data_S, data_E, data_I):
    """
    Término de pérdida de datos (Ldata): discrepancia entre predicción y observaciones.

    Adaptación de la ec. 3.6 del TFG para datos reales. Los pesos se normalizan
    por el rango al cuadrado (∆X²) de cada compartimento para que sus contribuciones
    al gradiente sean comparables, evitando que S (que varía ~50x más que I en
    términos de loss cuadrática) domine el entrenamiento.

    Solo se usan S e I como observables directos:
        · S: reconstruida con alta fiabilidad por conservación de masa.
        · I: convolución exponencial de la incidencia (C1, robusto).
        · E: NO se incluye como observable porque E = incidencia/κ es una
             derivada de los datos, y su ajuste produce residuales del ~59%
             en el pico, degradando la identificación de β(t). La ODE
             dI/dt = κE − γI la determina implícitamente a través de Lode.

    Máscara de pico (días 8–35, peso ×5):
        La epidemia tiene una fase de descenso larga (~60 días) con valores
        muy bajos. Sin la máscara, la suma de errores en la cola domina el
        promedio global y el optimizador aplana el pico para minimizar la cola.
        El factor ×5 fuerza a la red a priorizar el ajuste de la cresta de la ola.
    """
    S, E, I = pinnModel(model, data_t)

    # Pesos normalizados por rango al cuadrado (ec. 3.6 del TFG)
    w_S =  1.0 / ((S_base_np.max() - S_base_np.min())**2 + 1e-10)
    w_I = 50.0 / ((I_base_np.max() - I_base_np.min())**2 + 1e-10)

    # Máscara de pico: días 8–35 pesan ×5 para priorizar el ajuste del brote agudo
    t_vals      = data_t.view(-1)
    peak_mask   = ((t_vals >= 8.0) & (t_vals <= 35.0)).float().view(-1, 1)
    peak_weight = 1.0 + 4.0 * peak_mask   # factor total ×5 en el pico

    l_S = torch.mean((S - data_S)**2) * w_S
    l_I = torch.mean(peak_weight * (I - data_I)**2) * w_I
    return l_S + l_I

# ==============================================================================
# SECCIÓN E: EXPERIMENTO DE MONTE CARLO (50 runs, dropout 5%)
# ==============================================================================
#
# Protocolo de validación estadística para datos reales (Sección 3.2.3 del TFG):
# En cada iteración se descarta aleatoriamente el 5% de los días de datos y se
# varía la semilla de inicialización. El dropout simula la incertidumbre en la
# notificación oficial y la variabilidad en la convergencia del optimizador.
# La banda de incertidumbre de β(t) en las Figuras 3.2–3.3 se obtiene de aquí.
# ==============================================================================

n_ejecuciones       = 50
porcentaje_descarte = 0.05
total_dias          = len(t_base_np)
n_puntos_train      = int(total_dias * (1.0 - porcentaje_descarte))
num_epochs_Adam     = 5000
num_eval_points     = 1000

# Malla densa para evaluar β(t) y Rt después del entrenamiento
t_eval_grid = torch.linspace(0, T_MAX, num_eval_points).view(-1, 1).to(device)

# Matrices para almacenar resultados de las 50 iteraciones
historico_betas     = np.zeros((n_ejecuciones, num_eval_points))
lista_B_finales     = []
lista_alpha_finales = []

print(f"\n🚀 Iniciando experimento Monte Carlo (N={n_ejecuciones}) — Madrid\n")

for iteracion in range(n_ejecuciones):
    nueva_semilla = random_seed_value + iteracion
    torch.manual_seed(nueva_semilla)
    np.random.seed(nueva_semilla)

    # Dropout del 5%: selección aleatoria del 95% de los días para entrenamiento.
    # Simula la incertidumbre en la notificación oficial de casos.
    idx_train = np.sort(np.random.choice(
        np.arange(total_dias), size=n_puntos_train, replace=False
    ))

    data_t = torch.tensor(t_base_np[idx_train], device=device).view(-1, 1).requires_grad_(True)
    data_S = torch.tensor(S_base_np[idx_train], device=device).view(-1, 1)
    data_E = torch.tensor(E_base_np[idx_train], device=device).view(-1, 1)
    data_I = torch.tensor(I_base_np[idx_train], device=device).view(-1, 1)

    # Puntos de colocación para los residuos de las EDO (Latin Hypercube Sampling).
    # 810 puntos distribuidos uniformemente en [0, T_MAX] para cobertura homogénea.
    sampler = qmc.LatinHypercube(d=1)
    t_ode   = torch.tensor(
        sampler.random(n=810) * T_MAX,
        dtype=torch.float32, device=device, requires_grad=True
    )

    model = SEIR_PINN(t_base_np, m_base_np).to(device)
    model.net.apply(init_weights)

    # Optimizador Adam con tasas diferenciadas: lr más alto para α y φ porque
    # su espacio de búsqueda es más acotado que el de los pesos de la red.
    optimizer = torch.optim.Adam([
        {"params": model.net.parameters(),         "lr": 1e-4},
        {"params": [model.B_raw, model.alpha_raw], "lr": 1e-3},
    ])

    def get_weights(epoch):
        """
        Curriculum de física para datos reales: escalado progresivo de ωode.
        Se reduce el peso máximo de 1000 (modelo sintético) a 500 porque los
        datos reales tienen ruido y un peso de física demasiado alto produce
        sobreajuste a las EDO en detrimento del ajuste a los datos observados.
        """
        if   epoch <  500: return 1.0,   1.0
        elif epoch < 2000: return 1.0,  50.0
        elif epoch < 3500: return 1.0, 200.0
        else:              return 1.0, 500.0

    # --- Bucle de optimización Adam ---
    for epoch in range(num_epochs_Adam):
        w_data, w_ode = get_weights(epoch)
        optimizer.zero_grad()
        l_data = compute_loss_data(model, data_t, data_S, data_E, data_I)
        l_ode  = compute_loss_ode(model, t_ode)
        loss   = w_data * l_data + w_ode * l_ode
        loss.backward()
        # Gradient clipping: previene explosión de gradientes en datos ruidosos.
        # max_norm=1.0 limita la norma total del gradiente sin alterar su dirección.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # --- Refinamiento L-BFGS ---
    # lr=0.05 y history_size=20 más conservadores que en el modelo sintético
    # para evitar que L-BFGS sobreajuste la cola post-pico en datos reales.
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.05, max_iter=80,
        history_size=20, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        l_data = compute_loss_data(model, data_t, data_S, data_E, data_I)
        l_ode  = compute_loss_ode(model, t_ode)
        loss   = 500.0 * l_ode + 1.0 * l_data
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    # --- Registro de resultados ---
    with torch.no_grad():
        alpha_f    = torch.nn.functional.softplus(model.alpha_raw).item()
        B_f        = torch.nn.functional.softplus(model.B_raw).item()
        beta_curva = model.get_beta(t_eval_grid).cpu().numpy().flatten()

    historico_betas[iteracion, :] = beta_curva
    lista_B_finales.append(B_f)
    lista_alpha_finales.append(alpha_f)

    print(f"  [{iteracion+1:2d}/{n_ejecuciones}]  α={alpha_f:.4f}  "
          f"φ={B_f:.4f}  R₀≈{alpha_f/γ:.3f}")

# --- Informe global y exportación de β(t) ---
t_plot       = t_eval_grid.cpu().numpy().flatten()
media_beta_t = np.mean(historico_betas, axis=0)
std_beta_t   = np.std(historico_betas,  axis=0)
arr_B        = np.array(lista_B_finales)
arr_alpha    = np.array(lista_alpha_finales)

print(f"\n{'='*70}")
print(f"  INFORME GLOBAL (N={n_ejecuciones}, Madrid, Dropout {porcentaje_descarte*100:.0f}%)")
print(f"{'='*70}")
print(f"  α  : {np.mean(arr_alpha):.5f} ± {np.std(arr_alpha):.5f}")
print(f"  φ  : {np.mean(arr_B):.5f}     ± {np.std(arr_B):.5f}")
print(f"  R₀ : {np.mean(arr_alpha)/γ:.4f}   ± {np.std(arr_alpha)/γ:.4f}")

pd.DataFrame({
    't': t_plot,
    'beta_media': media_beta_t, 'beta_std': std_beta_t,
    'Rt_medio':  media_beta_t / γ, 'Rt_std': std_beta_t / γ,
}).to_csv('metricas_rt_madrid_v3.csv', index=False)
print("✓ CSV: metricas_rt_madrid_v3.csv")

# ==============================================================================
# SECCIÓN F: CÁLCULO DE RT Y REPORTE EPIDEMIOLÓGICO (TABLA 3.1 DEL TFG)
# ==============================================================================
#
# Rt = β(t) / γ. Se interpola β(t) de la malla de 1000 puntos a resolución
# diaria (días 0–90) para calcular los estadísticos mensuales de la Tabla 3.1.
# ==============================================================================

# Malla diaria exacta para métricas mensuales (días 0, 1, 2, ..., 90)
t_dias    = torch.linspace(0, 90, 91).view(-1, 1).to(device)
gamma_ref = γ   # Mismo valor que en todo el script

# Matriz Rt para las 50 iteraciones en resolución diaria
historico_rt = np.zeros((n_ejecuciones, 91))
t_eval_np    = t_eval_grid.cpu().numpy().flatten()

print(f"\n⏺ Calculando Rt diario (N={n_ejecuciones})...")

for iteracion in range(n_ejecuciones):
    # Interpolación lineal de β(t) desde la malla de 1000 pts a los 91 días exactos.
    # Necesaria porque historico_betas se calculó en la malla densa de entrenamiento.
    beta_diaria = np.interp(np.arange(91), t_eval_np, historico_betas[iteracion, :])
    historico_rt[iteracion, :] = beta_diaria / gamma_ref

# Estadísticos robustos por día
rt_media_diaria = np.mean(historico_rt, axis=0)
rt_std_diaria   = np.std(historico_rt,  axis=0)

df_rt = pd.DataFrame({
    'Dia':          np.arange(91),
    'Rt_Media':     rt_media_diaria,
    'Rt_Desviacion': rt_std_diaria
})

# Segmentación temporal por fases epidemiológicas (días desde t=0)
# Día 0 = 1 mar 2020; Día 30 = 31 mar; Día 60 = 30 abr; Día 90 = 30 may
marzo_m = df_rt.iloc[0:31]
abril_m  = df_rt.iloc[31:61]
mayo_m   = df_rt.iloc[61:91]

print(f"\n{'='*70}")
print(f"   REPORTE EPIDEMIOLÓGICO CONSOLIDADO: DINÁMICA DE Rt EN MADRID")
print(f"{'='*70}")
print(f" MARZO (Expansión inicial y confinamiento estricto):")
print(f"    Rt máximo           : {marzo_m['Rt_Media'].max():.2f} ± "
      f"{marzo_m['Rt_Desviacion'].iloc[marzo_m['Rt_Media'].argmax()]:.2f}")
print(f"    Rt medio mensual    : {marzo_m['Rt_Media'].mean():.2f} ± "
      f"{marzo_m['Rt_Desviacion'].mean():.2f}")
print(f"    Rt de cierre (día 30): {marzo_m['Rt_Media'].iloc[-1]:.2f} ± "
      f"{marzo_m['Rt_Desviacion'].iloc[-1]:.2f}")
print(f"{'-'*70}")
print(f" ABRIL (Cuarentena y mitigación del pico):")
print(f"    Rt medio mensual    : {abril_m['Rt_Media'].mean():.2f} ± "
      f"{abril_m['Rt_Desviacion'].mean():.2f}")
print(f"    Estado de control   : "
      f"{'Estabilizado (Rt < 1)' if abril_m['Rt_Media'].mean() < 1 else 'Alerta (Rt > 1)'}")
print(f"{'-'*70}")
print(f" MAYO (Inicio de desescalada):")
print(f"    Rt medio mensual    : {mayo_m['Rt_Media'].mean():.2f} ± "
      f"{mayo_m['Rt_Desviacion'].mean():.2f}")
print(f"    Rt final (día 90)   : {mayo_m['Rt_Media'].iloc[-1]:.2f} ± "
      f"{mayo_m['Rt_Desviacion'].iloc[-1]:.2f}")
print(f"{'='*70}")

df_rt.to_csv("metricas_rt_madrid_montecarlo.csv", index=False)
print("✓ CSV: metricas_rt_madrid_montecarlo.csv")

# ==============================================================================
# SECCIÓN G: VISUALIZACIÓN — FIGURA 3.2 DEL TFG
# ==============================================================================
#
# Panel de 5 subgráficos (a–d compartimentos SEIR + e β(t)) con:
#   · Eje X en fechas reales (no días) para los paneles a–d
#   · Compartimentos E, I, R escalados a ‰ de la población
#   · Banda de incertidumbre ±σ del experimento Monte Carlo en β(t)
#   · Líneas verticales con eventos epidemiológicos clave
# ==============================================================================

plt.rcParams['pdf.fonttype'] = 42   # Fuentes vectoriales Type 42 para EPS/PDF
plt.rcParams['ps.fonttype']  = 42
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def t_a_fecha(t_array):
    """Convierte días desde t=0 a objetos datetime para el eje X."""
    return [FECHA_INICIO + timedelta(days=float(t)) for t in t_array]


def _add_eventos(ax, usar_fechas=True):
    """Añade líneas verticales para los eventos epidemiológicos clave."""
    colores_ev = ['#e41a1c', '#984ea3', '#377eb8']
    for (label, t_ev), c_ev in zip(EVENTOS.items(), colores_ev):
        x_ev = FECHA_INICIO + timedelta(days=t_ev) if usar_fechas else t_ev
        ax.axvline(x_ev, color=c_ev, linestyle='--', lw=1.2, alpha=0.65,
                   label=label)


def _fmt_eje_fecha(ax):
    """Formatea el eje X con etiquetas de fecha cada dos semanas."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=14)
    ax.set_xlabel('Fecha (2020)', fontsize=14)


print("\n⏺ Generando Figura 3.2: panel de resultados PINN — Madrid...")

# Malla densa para la predicción continua de los compartimentos
t_eval = torch.linspace(0, T_MAX, 1000).view(-1, 1).to(device)

with torch.no_grad():
    S_pred, E_pred, I_pred = pinnModel(model, t_eval)
    R_pred = 1.0 - (S_pred + E_pred + I_pred)

def to_np(x):
    return x.detach().cpu().numpy().flatten()

scale_EIR   = 1e3   # Escala a ‰ de la población para E, I, R
t_eval_np   = to_np(t_eval)
fechas_eval = t_a_fecha(t_eval_np)
fechas_data = t_a_fecha(to_np(data_t))

# Estadísticos Monte Carlo de β(t) ya calculados en la Sección E
media_beta = media_beta_t
std_beta   = std_beta_t

# Datos observados (última iteración, representativa del experimento)
obs = {
    'S': to_np(data_S),
    'E': to_np(data_E) * scale_EIR,
    'I': to_np(data_I) * scale_EIR,
    'R': (1.0 - to_np(data_S) - to_np(data_E) - to_np(data_I)) * scale_EIR,
}
pred = {
    'S': to_np(S_pred),
    'E': to_np(E_pred) * scale_EIR,
    'I': to_np(I_pred) * scale_EIR,
    'R': to_np(R_pred) * scale_EIR,
}

colors = {'S': '#1f77b4', 'E': '#ff7f0e', 'I': '#d62728', 'R': '#2ca02c'}
titulos = {
    'S': 'a) S — Susceptibles',
    'E': 'b) E — Expuestos',
    'I': 'c) I — Infectados',
    'R': 'd) R — Recuperados',
}
etiq_y = {
    'S': 'Fracción susceptible',
    'E': 'Población (‰)', 'I': 'Población (‰)', 'R': 'Población (‰)',
}

fig = plt.figure(figsize=(18, 15))
gs  = fig.add_gridspec(3, 2)
axes_seir = {
    'S': fig.add_subplot(gs[0, 0]),
    'E': fig.add_subplot(gs[0, 1]),
    'I': fig.add_subplot(gs[1, 0]),
    'R': fig.add_subplot(gs[1, 1]),
}
ax_beta = fig.add_subplot(gs[2, :])

# --- Paneles a–d: compartimentos SEIR ---
for clave, ax in axes_seir.items():
    c = colors[clave]
    ax.scatter(fechas_data, obs[clave],
               color=c, s=20, alpha=0.35, zorder=3, label='Datos observados')
    ax.plot(fechas_eval, pred[clave],
            color=c, lw=2.5, zorder=4, label='Ajuste PINN')
    _add_eventos(ax, usar_fechas=True)
    ax.set_title(titulos[clave], fontsize=18, fontweight='bold', loc='left')
    ax.set_ylabel(etiq_y[clave], fontsize=14)
    ax.grid(True, alpha=0.20)
    _fmt_eje_fecha(ax)
    legend_loc = 'upper right' if clave != 'R' else 'lower right'
    ax.legend(loc=legend_loc, fontsize=13, framealpha=0.85)

# --- Panel e: β(t) con incertidumbre Monte Carlo ---
ax_beta.plot(t_plot, media_beta,
             color='purple', lw=3.5, label=r'$\beta(t)$ Media Identificada')
ax_beta.plot(t_plot, media_beta + std_beta,
             color='mediumpurple', lw=1.5, ls='--',
             label=r'Incertidumbre acotada ($\pm\sigma$)')
ax_beta.plot(t_plot, media_beta - std_beta,
             color='mediumpurple', lw=1.5, ls='--')
ax_beta.fill_between(t_plot, media_beta - std_beta, media_beta + std_beta,
                     color='mediumpurple', alpha=0.15)
_add_eventos(ax_beta, usar_fechas=False)
ax_beta.set_title(r'e) Inferencia de la función de transmisión $\beta(t)$',
                  fontsize=20, fontweight='bold', color='purple', loc='left')
ax_beta.set_xlabel('Tiempo (días desde el origen epidemiológico)', fontsize=16)
ax_beta.set_ylabel(r'Tasa de Transmisión $\beta(t)$', fontsize=16)
ax_beta.set_xlim(0, T_MAX)
ax_beta.grid(True, linestyle='--', alpha=0.5)
ax_beta.legend(loc='upper right', fontsize=14)

fig.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.25)
fig.savefig('resultados_pinn_madrid.eps', format='eps', bbox_inches='tight')
fig.savefig('resultados_pinn_madrid.pdf', format='pdf', bbox_inches='tight')
plt.show()
print("✓ Figura guardada: resultados_pinn_madrid.eps / .pdf")

# ==============================================================================
# SECCIÓN H: FIGURA COMPARATIVA Rt — MADRID vs TOLEDO (Figura 3.4)
# ==============================================================================
#
# Genera el panel comparativo del número reproductivo efectivo Rt(t) para
# la Comunidad de Madrid y la provincia de Toledo durante la primera ola.
#
# Requisito previo: haber ejecutado también PINN_Toledo.py para disponer
# del fichero metricas_rt_toledo_v3.csv en el mismo directorio de trabajo.
# Ambos CSVs se generan al final de la Sección E de sus respectivos scripts.
#
# Salidas:
#   · comparativa_rt_madrid_toledo.eps/.pdf : Figura 3.4 del TFG
# ==============================================================================

import matplotlib.dates as mdates

FECHA_INICIO_COMP = datetime(2020, 3, 1)

EVENTOS_COMP = {
    'Estado de alarma (14 mar)':  13,
    'Fin confinamiento estricto': 43,
    'Inicio desescalada (1 may)': 61,
}

# Carga de los CSVs de β(t) y Rt generados por ambas PINNs
df_M = pd.read_csv('metricas_rt_madrid_v3.csv')
df_T = pd.read_csv('metricas_rt_toledo_v3.csv')

fechas_M   = [FECHA_INICIO_COMP + timedelta(days=float(d)) for d in df_M['t']]
fechas_T   = [FECHA_INICIO_COMP + timedelta(days=float(d)) for d in df_T['t']]
rt_media_M = df_M['Rt_medio'].values
rt_std_M   = df_M['Rt_std'].values
rt_media_T = df_T['Rt_medio'].values
rt_std_T   = df_T['Rt_std'].values

# σ máximo de cada provincia: se incluye en la leyenda para cuantificar
# la incertidumbre del experimento Monte Carlo (véase Tabla 3.3 del TFG).
sigma_max_M = rt_std_M.max()
sigma_max_T = rt_std_T.max()

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(fechas_M, rt_media_M,
        color='#1f77b4', lw=2.5,
        label=fr'$R_t$ Madrid ($\sigma_{{max}}={sigma_max_M:.2f}$)')
ax.plot(fechas_T, rt_media_T,
        color='#d62728', lw=2.5, linestyle='--',
        label=fr'$R_t$ Toledo ($\sigma_{{max}}={sigma_max_T:.2f}$)')

# Umbral de control epidémico
ax.axhline(1.0, color='black', linestyle=':', lw=1.5, alpha=0.7,
           label=r'Umbral crítico ($R_t = 1$)')

# Líneas verticales de eventos epidemiológicos clave
colores_ev = ['#e41a1c', '#984ea3', '#377eb8']
for (label, t_ev), c_ev in zip(EVENTOS_COMP.items(), colores_ev):
    ax.axvline(FECHA_INICIO_COMP + timedelta(days=t_ev),
               color=c_ev, linestyle='--', lw=1.2, alpha=0.65, label=label)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=14)
ax.set_xlabel('Fecha (2020)', fontsize=15)
ax.set_ylabel(r'Número reproductivo efectivo $R_t$', fontsize=15)
ax.set_ylim(bottom=0)
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(loc='upper right', fontsize=15, framealpha=0.90, ncol=2)

fig.tight_layout()
fig.savefig('comparativa_rt_madrid_toledo.eps', format='eps', bbox_inches='tight')
fig.savefig('comparativa_rt_madrid_toledo.pdf', format='pdf', bbox_inches='tight')
plt.show()
print("✓ Figura 3.4 guardada: comparativa_rt_madrid_toledo.eps / .pdf")