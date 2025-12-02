import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import StringIO

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Diseño Avanzado de Micropilotes", layout="wide", page_icon="🏗️")

# Estilos CSS para pestañas
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; border-bottom: 2px solid #FF4B4B; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Sistema de Diseño de Micropilotes")
st.markdown("Optimización de diseño, análisis de huella de carbono y transferencia de carga.")

tab_diseno, tab_geo = st.tabs(["📐 Diseño & Optimización", "🌍 Correlaciones Geotécnicas (SPT)"])

# ==============================================================================
# PARÁMETROS FIJOS GLOBALES
# ==============================================================================
DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85}
LISTA_D = sorted(list(DIAMETROS_COM.keys()))
MIN_MICROS = 3
MAX_MICROS = 15
RANGO_L = range(5, 36) 
COSTO_PERF_BASE = 100

# Factores Ambientales
FACTOR_CO2_CEMENTO = 0.90
FACTOR_CO2_PERF = 15.0
FACTOR_CO2_ACERO = 1.85
DENSIDAD_ACERO = 7850.0
DENSIDAD_CEMENTO = 3150.0
FY_ACERO_KPA = 500000.0 # 500 MPa

COLORES_ESTRATOS = ["#D7BDE2", "#A9CCE3", "#A3E4D7", "#F9E79F", "#F5B7B1", "#D2B4DE", "#AED6F1", "#A2D9CE"]

# ==============================================================================
# ---------------------- PESTAÑA 1: DISEÑO & OPTIMIZACIÓN ----------------------
# ==============================================================================
with tab_diseno:
    # 1. BARRA LATERAL
    with st.sidebar:
        st.header("1. Definición de Estratigrafía")
        num_capas = st.slider("Número de Capas", 1, 8, 3)
        
        ESTRATOS = []
        z_acumulada = 0.0
        
        with st.expander("Configurar Capas", expanded=True):
            for i in range(num_capas):
                st.markdown(f"**--- Capa {i+1} ---**")
                c1, c2, c3 = st.columns(3)
                espesor = c1.number_input(f"Espesor (m)", value=3.0 if i==0 else 5.0, step=0.5, key=f"e{i}")
                qs = c2.number_input(f"Adherencia qs (kPa)", value=40.0 + i*20.0, step=5.0, key=f"q{i}")
                f_exp = c3.number_input(f"Fact. Exp.", value=1.1 + i*0.05, step=0.1, key=f"f{i}")
                
                z_acumulada += espesor
                ESTRATOS.append({
                    "espesor": espesor, "z_fin": z_acumulada, "qs": qs, "f_exp": f_exp,
                    "color": COLORES_ESTRATOS[i % len(COLORES_ESTRATOS)]
                })

        st.caption(f"Prof. Total: {z_acumulada:.1f} m")

        st.header("2. Cargas y Seguridad")
        CARGA_TON = st.number_input("Carga Cabezal (Ton)", value=109.0, step=1.0)
        FS_REQ = st.slider("Factor de Seguridad", 1.5, 3.0, 2.0, 0.1)
        
        st.header("3. Materiales")
        RELACION_AGUA_CEMENTO = st.slider("Relación A/C", 0.4, 0.6, 0.50, 0.05)
        
        st.divider()
        calcular = st.button("🚀 CALCULAR DISEÑO", type="primary")

    # 2. FUNCIONES DE CÁLCULO
    def calc_capacidad_individual_dinamica(D, L_total, estratos_data):
        Q_ult = 0; z_actual = 0
        for estrato in estratos_data:
            if z_actual >= L_total: break
            z_fondo = min(estrato["z_fin"], L_total)
            espesor = z_fondo - z_actual
            if espesor > 0:
                Q_ult += (math.pi * D * espesor) * estrato["qs"]
            z_actual = z_fondo
        return Q_ult

    def calc_volumenes_grout(D, L_total, estratos_data):
        vol_teo = 0; vol_exp = 0; z_actual = 0
        area = math.pi * (D/2)**2
        for estrato in estratos_data:
            if z_actual >= L_total: break
            z_fondo = min(estrato["z_fin"], L_total)
            espesor = z_fondo - z_actual
            if espesor > 0:
                v_t = area * espesor
                vol_teo += v_t
                vol_exp += v_t * estrato["f_exp"]
            z_actual = z_fondo
        return vol_teo, vol_exp

    def get_peso_cemento_m3(wc):
        return 1.0 / (wc/1000.0 + 1.0/DENSIDAD_CEMENTO)

    def calc_impacto_ambiental(L_total, N, vol_grout_exp, Q_act_kpa):
        area_acero = Q_act_kpa / FY_ACERO_KPA
        vol_acero = area_acero * L_total * N
        peso_acero = vol_acero * DENSIDAD_ACERO
        
        vol_grout_neto = max(0, vol_grout_exp - vol_acero)
        peso_cemento = vol_grout_neto * get_peso_cemento_m3(RELACION_AGUA_CEMENTO)
        
        co2 = (peso_acero*FACTOR_CO2_ACERO + peso_cemento*FACTOR_CO2_CEMENTO + (L_total*N)*FACTOR_CO2_PERF) / 1000
        return co2, area_acero * 10000

    def obtener_perfil_resistencia(D, L, estratos_data):
        z_points = [0]; q_points = [0]; z_act = 0; q_acum = 0
        for estrato in estratos_data:
            if z_act >= L: break
            z_end = min(estrato["z_fin"], L)
            esp = z_end - z_act
            if esp > 0:
                q_acum += (math.pi * D * esp) * estrato["qs"]
                z_points.append(z_end); q_points.append(q_acum)
            z_act = z_end
        return z_points, q_points

    # 3. EJECUCIÓN
    if calcular:
        CARGA_REQ_KN = CARGA_TON * 9.81
        resultados_raw = []

        with st.spinner('Calculando...'):
            for D in LISTA_D:
                for N in range(MIN_MICROS, MAX_MICROS + 1):
                    Q_act_kn = CARGA_REQ_KN / N
                    Q_req_geo = Q_act_kn * FS_REQ
                    
                    for L in RANGO_L:
                        Q_ult = calc_capacidad_individual_dinamica(D, L, ESTRATOS)
                        
                        if Q_ult >= Q_req_geo:
                            FS = Q_ult / Q_act_kn
                            v_teo, v_exp = calc_volumenes_grout(D, L, ESTRATOS)
                            v_teo_tot = v_teo * N; v_exp_tot = v_exp * N
                            
                            eff = DIAMETROS_COM[D]
                            costo = (L * N * COSTO_PERF_BASE) / eff
                            
                            co2, acero_cm2 = calc_impacto_ambiental(L, N, v_exp_tot, Q_act_kn)
                            
                            resultados_raw.append({
                                "D_val": D, "D_mm": int(D*1000), "N": N, "L_m": L, "L_Tot_m": L*N,
                                "FS": FS, "Vol_Teo": v_teo_tot, "Vol_Exp": v_exp_tot,
                                "Costo_Idx": costo, "Eficiencia": eff, "CO2_ton": co2,
                                "Q_adm": Q_ult/FS_REQ/9.81, "Q_act": Q_act_kn/9.81,
                                "Acero_cm2": acero_cm2
                            })
                            break 

        if not resultados_raw:
            st.error("No se encontraron soluciones.")
        else:
            # Selección Top 10 Variada
            resultados_raw.sort(key=lambda x: x["Costo_Idx"])
            top_10 = []
            ids = set()
            
            for d_target in LISTA_D:
                c = next((r for r in resultados_raw if r["D_val"] == d_target), None)
                if c:
                    uid = f"{c['D_mm']}-{c['N']}-{c['L_m']}"
                    if uid not in ids: top_10.append(c); ids.add(uid)
            
            for r in resultados_raw:
                if len(top_10) >= 10: break
                uid = f"{r['D_mm']}-{r['N']}-{r['L_m']}"
                if uid not in ids: top_10.append(r); ids.add(uid)

            top_10.sort(key=lambda x: x["Costo_Idx"])
            df = pd.DataFrame(top_10)
            best = df.iloc[0]

            # --- DASHBOARD ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mejor Opción", f"{int(best['N'])} x Ø{int(best['D_mm'])}mm")
            c2.metric("Longitud", f"{int(best['L_m'])} m")
            c3.metric("Volumen Grout", f"{best['Vol_Exp']:.1f} m³")
            c4.metric("Huella CO2", f"{best['CO2_ton']:.1f} Ton")
            st.divider()
            
            col_graf, col_tabla = st.columns([1, 1.2])
            
            with col_graf:
                st.subheader("📉 Transferencia de Carga")
                fig_prof, ax_prof = plt.subplots(figsize=(8, 6))
                
                # Fondo estratos (Rectángulos muy anchos para cubrir todo)
                y_lim_plot = max([r['L_m'] for r in top_10]) + 2
                prev_z = 0
                for e in ESTRATOS:
                    h_layer = min(e["z_fin"], y_lim_plot) - prev_z
                    if h_layer > 0:
                        rect = patches.Rectangle((0, prev_z), 10000, h_layer, color=e["color"], alpha=0.3)
                        ax_prof.add_patch(rect)
                        ax_prof.text(1, prev_z + h_layer/2, f"qs={int(e['qs'])}", color='#555', va='center', fontsize=8)
                    prev_z = e["z_fin"]
                    if prev_z >= y_lim_plot: break

                # Curvas
                diametros_vistos = set()
                plotted_count = 0
                max_x_val = 0 # Variable para ajuste dinámico del eje X
                
                for i, row in df.iterrows():
                    if row['D_mm'] in diametros_vistos and plotted_count > 2: continue
                    
                    z_p, q_p = obtener_perfil_resistencia_grafico(row['D_val'], row['L_m'], ESTRATOS)
                    q_adm_graf = [(q / FS_REQ) / 9.81 for q in q_p]
                    
                    # Rastrear máximo para el eje X
                    if max(q_adm_graf) > max_x_val: max_x_val = max(q_adm_graf)
                    
                    label_str = f"Ø{row['D_mm']} (Qadm: {q_adm_graf[-1]:.0f}T)"
                    ax_prof.plot(q_adm_graf, z_p, marker='o', markersize=4, linewidth=2, label=label_str, alpha=0.8 if i==0 else 0.5)
                    diametros_vistos.add(row['D_mm'])
                    plotted_count +=1

                ax_prof.set_ylim(y_lim_plot, 0)
                
                # --- AJUSTE DINÁMICO DEL EJE X ---
                if max_x_val > 0:
                    ax_prof.set_xlim(0, max_x_val * 1.15) # +15% de margen
                
                ax_prof.set_ylabel("Profundidad (m)")
                ax_prof.set_xlabel("Capacidad Admisible Acumulada (Ton)")
                ax_prof.legend(loc='lower right', fontsize=8)
                ax_prof.grid(True, linestyle=':', alpha=0.5)
                st.pyplot(fig_prof)

            with col_tabla:
                st.subheader("📋 Top 10 Alternativas")
                df_show = df[["D_mm", "N", "L_m", "L_Tot_m", "FS", "Q_adm", "Q_act", "Vol_Exp", "CO2_ton"]].copy()
                df_show.columns = ["Ø(mm)", "Cant", "L(m)", "Perf(m)", "FS", "Qadm", "Qact", "Grout", "CO2"]
                st.dataframe(
                    df_show.style.background_gradient(subset=["Perf(m)"], cmap="Blues_r")
                           .background_gradient(subset=["CO2"], cmap="Greens_r")
                           .format("{:.1f}", subset=["Qadm", "Qact", "Grout", "CO2"])
                           .format("{:.2f}", subset=["FS"]),
                    use_container_width=True, height=400
                )
    else:
        st.info("👈 Configure estratos y presione calcular.")

# ==============================================================================
# ---------------------- PESTAÑA 2: CORRELACIONES SPT --------------------------
# ==============================================================================
with tab_geo:
    st.header("🌍 Correlación SPT -> Adherencia")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.write("Pegue sus datos (Profundidad, N):")
        spt_input = st.text_area("Datos", value="1.5, 4\n3.0, 7\n4.5, 12\n6.0, 15\n9.0, 28\n12.0, 42", height=200)
        k_val = st.slider("Factor K (qs = K * N)", 1.0, 6.0, 3.5, 0.1)
        proc_spt = st.button("Graficar Correlación")
    
    with c2:
        st.markdown("""
        **Guía FHWA (Grout Tipo A):**
        * Arcillas/Limos: $q_s = 20-90$ kPa
        * Arenas Densas: $q_s = 100-250$ kPa
        """)
    
    if proc_spt and spt_input:
        try:
            df_spt = pd.read_csv(StringIO(spt_input), names=["z", "N"], header=None)
            df_spt["qs"] = df_spt["N"] * k_val
            
            fig_spt, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
            ax1.plot(df_spt["N"], df_spt["z"], 'b-o'); ax1.set_title("N-SPT"); ax1.invert_yaxis(); ax1.grid(True)
            ax2.plot(df_spt["qs"], df_spt["z"], 'r-s'); ax2.set_title(f"Adherencia Estimada (K={k_val})"); ax2.grid(True)
            st.pyplot(fig_spt)
        except: st.error("Formato incorrecto. Use: Profundidad, N")
