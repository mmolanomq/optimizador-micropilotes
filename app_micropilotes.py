import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Diseño Micropilotes + Gráficas de Transferencia", layout="wide", page_icon="🏗️")

st.title("🏗️ Optimizador de Micropilotes (Análisis de Transferencia de Carga)")
st.markdown("""
Encuentra la configuración óptima y visualiza **cómo aporta cada estrato** a la resistencia total.
""")

# ==============================================================================
# 1. BARRA LATERAL (INPUTS)
# ==============================================================================
with st.sidebar:
    st.header("1. Estratigrafía")
    with st.expander("Definir Capas de Suelo", expanded=True):
        c1, c2 = st.columns(2)
        z1 = c1.number_input("Prof. E1 (m)", value=3.0)
        qs1 = c2.number_input("Adherencia E1 (kPa)", value=40.0)
        
        c3, c4 = st.columns(2)
        z2 = c3.number_input("Prof. E2 (m)", value=8.0)
        qs2 = c4.number_input("Adherencia E2 (kPa)", value=60.0)
        
        c5, c6 = st.columns(2)
        z3 = c5.number_input("Prof. E3 (m)", value=15.0)
        qs3 = c6.number_input("Adherencia E3 (kPa)", value=100.0)
        
        c7, c8 = st.columns(2)
        z4 = c7.number_input("Prof. E4 (m)", value=30.0)
        qs4 = c8.number_input("Adherencia E4 (kPa)", value=150.0)

    ESTRATOS = [
        {"z_fin": z1, "qs": qs1, "color": "#D7BDE2"},
        {"z_fin": z2, "qs": qs2, "color": "#A9CCE3"},
        {"z_fin": z3, "qs": qs3, "color": "#A3E4D7"},
        {"z_fin": z4, "qs": qs4, "color": "#F9E79F"}
    ]

    st.header("2. Cargas y Seguridad")
    CARGA_TON = st.number_input("Carga (Ton)", value=109.0)
    FS_REQ = st.slider("Factor de Seguridad", 1.5, 3.0, 2.0, 0.1)
    
    st.header("3. Materiales")
    RELACION_AGUA_CEMENTO = st.slider("A/C Lechada", 0.4, 0.6, 0.50, 0.05)
    FACTOR_EXPANSION = st.number_input("Factor Expansión", value=1.2)
    
    col_acero1, col_acero2 = st.columns(2)
    with col_acero1: NUM_BARRAS = st.number_input("No. Barras", value=1)
    with col_acero2: DIAM_BARRA = st.selectbox("Ø Barra (mm)", [25, 32, 40, 57], index=1)

    st.divider()
    calcular = st.button("🚀 CALCULAR Y GRAFICAR", type="primary")

# ==============================================================================
# 2. PARÁMETROS FIJOS
# ==============================================================================
DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.130: 0.90, 0.150: 0.85, 0.200: 0.80}
LISTA_D = sorted(list(DIAMETROS_COM.keys()))
MIN_MICROS = 3
MAX_MICROS = 20 # Aumentado para asegurar soluciones
RANGO_L = range(6, 31) # Buscamos hasta 30m
COSTO_PERF_BASE = 100

# Factores Ambientales
FACTOR_CO2_CEMENTO, FACTOR_CO2_PERF, FACTOR_CO2_ACERO = 0.90, 15.0, 1.85
DENSIDAD_ACERO, DENSIDAD_CEMENTO = 7850.0, 3150.0

# ==============================================================================
# 3. FUNCIONES DE CÁLCULO
# ==============================================================================
def calc_capacidad_individual(D, L, estratos_data):
    Q_ult = 0; z_actual = 0
    for estrato in estratos_data:
        if z_actual >= L: break
        z_fondo = min(estrato["z_fin"], L)
        espesor = z_fondo - z_actual
        if espesor > 0:
            Q_ult += (math.pi * D * espesor) * estrato["qs"]
        z_actual = z_fondo
    return Q_ult

def obtener_perfil_resistencia(D, L, estratos_data):
    """Genera los puntos (profundidad, capacidad_acumulada) para graficar"""
    z_points = [0]
    q_points = [0]
    z_actual = 0
    q_acum = 0
    
    # Iteramos por estratos para obtener puntos de quiebre exactos
    for estrato in estratos_data:
        if z_actual >= L: break
        
        z_fondo_tramo = min(estrato["z_fin"], L)
        espesor = z_fondo_tramo - z_actual
        
        if espesor > 0:
            # Aporte de este tramo
            q_tramo = (math.pi * D * espesor) * estrato["qs"]
            q_acum += q_tramo
            
            # Guardar punto
            z_points.append(z_fondo_tramo)
            q_points.append(q_acum)
        
        z_actual = z_fondo_tramo
        
    return z_points, q_points

def get_peso_cemento_m3(wc):
    return 1.0 / (wc/1000.0 + 1.0/DENSIDAD_CEMENTO)

def calc_impacto_ambiental(D, L, N, vol_perf_exp):
    area_acero = NUM_BARRAS * (math.pi * (DIAM_BARRA/1000)**2 / 4)
    vol_acero = area_acero * L * N
    peso_acero = vol_acero * DENSIDAD_ACERO
    vol_grout_real = max(0, vol_perf_exp - vol_acero)
    peso_cemento = vol_grout_real * get_peso_cemento_m3(RELACION_AGUA_CEMENTO)
    return (peso_acero*FACTOR_CO2_ACERO + peso_cemento*FACTOR_CO2_CEMENTO + (L*N)*FACTOR_CO2_PERF) / 1000

# ==============================================================================
# 4. LÓGICA PRINCIPAL
# ==============================================================================
if calcular:
    CARGA_REQ = CARGA_TON * 9.81
    resultados_raw = []

    with st.spinner('Calculando perfiles de resistencia...'):
        for D in LISTA_D:
            if (D*1000) < (DIAM_BARRA + 40): continue 

            for N in range(MIN_MICROS, MAX_MICROS + 1):
                found_for_N = False
                for L in RANGO_L:
                    Q_ind = calc_capacidad_individual(D, L, ESTRATOS)
                    Q_grupo = Q_ind * N
                    FS = Q_grupo / CARGA_REQ
                    
                    if FS >= FS_REQ:
                        # Cálculos Económicos y Ecológicos
                        eficiencia = DIAMETROS_COM[D]
                        costo_idx = (L * N * COSTO_PERF_BASE) / eficiencia
                        vol_teo = math.pi * (D/2)**2 * L * N
                        vol_exp = vol_teo * FACTOR_EXPANSION
                        co2 = calc_impacto_ambiental(D, L, N, vol_exp)
                        
                        resultados_raw.append({
                            "D_val": D, "D_mm": int(D*1000),
                            "N": N, "L_m": L, "L_Tot_m": L * N,
                            "FS": FS, "Vol_Exp": vol_exp,
                            "Costo_Idx": costo_idx, "Eficiencia": eficiencia,
                            "CO2_ton": co2,
                            "Q_adm": (Q_ind/FS_REQ)/9.81,
                            "Q_act": (CARGA_REQ/N)/9.81
                        })
                        found_for_N = True
                        break # L mínima para este N
                
                # Si no encontramos L < 30m para este N, seguimos probando con N+1

    # --- SELECCIÓN TOP 10 ---
    if not resultados_raw:
        st.error("No se encontraron soluciones. Aumente la cantidad de micropilotes o mejore el suelo.")
    else:
        # Ordenar por costo
        resultados_raw.sort(key=lambda x: x["Costo_Idx"])
        
        # Estrategia para garantizar 10 opciones diversas
        top_10 = []
        ids = set()
        
        # 1. Asegurar al menos 1 de cada diámetro
        for d_target in LISTA_D:
            cand = next((r for r in resultados_raw if r["D_val"] == d_target), None)
            if cand:
                uid = f"{cand['D_mm']}-{cand['N']}-{cand['L_m']}"
                if uid not in ids: top_10.append(cand); ids.add(uid)
        
        # 2. Rellenar hasta 10 con los mejores restantes
        for r in resultados_raw:
            if len(top_10) >= 10: break
            uid = f"{r['D_mm']}-{r['N']}-{r['L_m']}"
            if uid not in ids: top_10.append(r); ids.add(uid)

        # Reordenar por costo para la tabla final
        top_10.sort(key=lambda x: x["Costo_Idx"])
        df = pd.DataFrame(top_10)

        # --- MOSTRAR RESULTADOS ---
        
        # 1. KPIs Principales
        best = df.iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Mejor Opción (Costo)", f"{int(best['N'])} micros de Ø{int(best['D_mm'])}mm", f"L={int(best['L_m'])}m")
        k2.metric("Huella CO2", f"{best['CO2_ton']:.2f} Ton", f"FS={best['FS']:.2f}")
        k3.metric("Perforación Total", f"{best['L_Tot_m']} m", f"Grout: {best['Vol_Exp']:.1f}m³")

        st.divider()
        
        # 2. GRÁFICA DE TRANSFERENCIA DE CARGA (NUEVA)
        st.subheader("📉 Análisis de Transferencia de Carga (Carga vs Profundidad)")
        st.markdown("Esta gráfica muestra cómo **un solo micropilote** gana capacidad de carga a medida que profundiza.")
        
        fig_prof, ax_prof = plt.subplots(figsize=(10, 6))
        
        # Dibujar zonas de estratos en el fondo
        y_lim_plot = max([r['L_m'] for r in top_10]) + 2
        prev_z = 0
        for e in ESTRATOS:
            h_layer = min(e["z_fin"], y_lim_plot) - prev_z
            if h_layer > 0:
                rect = patches.Rectangle((0, prev_z), 500, h_layer, color=e["color"], alpha=0.3)
                ax_prof.add_patch(rect)
                # Etiqueta del estrato
                ax_prof.text(5, prev_z + h_layer/2, f"qs={int(e['qs'])} kPa", color='gray', va='center', fontsize=9)
            prev_z = e["z_fin"]
            if prev_z >= y_lim_plot: break

        # Graficar las curvas de las mejores opciones (Top 3 para no saturar)
        colors = ['#E74C3C', '#2980B9', '#27AE60', '#8E44AD']
        plotted_configs = 0
        
        # Agrupamos por diámetro para mostrar 1 curva representativa por diámetro
        diametros_vistos = set()
        
        for i, row in df.iterrows():
            if row['D_mm'] in diametros_vistos: continue
            
            # Calcular perfil
            z_p, q_p = obtener_perfil_resistencia(row['D_val'], row['L_m'], ESTRATOS)
            
            # Convertir Q última a Q admisible para la gráfica (más útil para diseño)
            q_p_adm_ton = [(q / FS_REQ) / 9.81 for q in q_p]
            
            label_str = f"Ø{row['D_mm']}mm (Qadm final: {q_p_adm_ton[-1]:.1f} Ton)"
            ax_prof.plot(q_p_adm_ton, z_p, marker='o', markersize=4, linewidth=2, label=label_str)
            
            # Marcar la longitud de diseño
            ax_prof.hlines(row['L_m'], 0, q_p_adm_ton[-1], colors='gray', linestyles='dotted')
            
            diametros_vistos.add(row['D_mm'])
            plotted_configs += 1
            if plotted_configs >= 4: break # Máximo 4 curvas

        ax_prof.set_ylim(y_lim_plot, 0) # Invertir eje Y
        ax_prof.set_xlim(0, None)
        ax_prof.set_ylabel("Profundidad (m)")
        ax_prof.set_xlabel("Capacidad Admisible Acumulada por Micropilote (Ton)")
        ax_prof.set_title("Evolución de la Resistencia por Fricción con la Profundidad")
        ax_prof.legend(loc='lower right')
        ax_prof.grid(True, linestyle=':', alpha=0.5)
        
        st.pyplot(fig_prof)
        st.caption("Nota: Las áreas de fondo representan los estratos. Observe cómo la pendiente de la curva aumenta en estratos con mayor adherencia (qs).")

        st.divider()

        # 3. TABLA DETALLADA (TOP 10)
        st.subheader("📋 Top 10 Alternativas de Diseño")
        
        df_show = df[["D_mm", "N", "L_m", "L_Tot_m", "FS", "Q_adm", "Q_act", "Vol_Exp", "CO2_ton"]].copy()
        df_show.columns = ["Ø (mm)", "Cant.", "L (m)", "Perf. Total", "FS", "Q adm", "Q act", "Grout (m³)", "CO2 (Ton)"]
        
        st.dataframe(
            df_show.style.background_gradient(subset=["Perf. Total"], cmap="Blues")
                   .background_gradient(subset=["CO2 (Ton)"], cmap="Greens")
                   .format("{:.2f}", subset=["FS", "Q adm", "Q act", "Grout (m³)", "CO2 (Ton)"]),
            use_container_width=True
        )

else:
    st.info("👈 Configure los estratos y presione el botón para optimizar.")