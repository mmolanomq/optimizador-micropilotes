import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import StringIO
import re # Para validar email

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import datetime

import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import StringIO
import re # Para validar email

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Diseño Avanzado de Micropilotes", layout="wide", page_icon="🏗️")

# ==============================================================================
# 0. SISTEMA DE REGISTRO (EL MURO)
# ==============================================================================
# Inicializar estado de sesión si no existe
if 'usuario_registrado' not in st.session_state:
    st.session_state['usuario_registrado'] = False

def mostrar_registro():
    """Muestra la pantalla de bloqueo/registro"""
    st.markdown("## 🔒 Acceso a Herramienta de Ingeniería")
    st.info("Para acceder a la calculadora de optimización y huella de carbono, por favor regístrese.")
    
    with st.form("formulario_registro"):
        col1, col2 = st.columns(2)
        nombre = col1.text_input("Nombre Completo")
        empresa = col2.text_input("Empresa / Universidad")
        email = st.text_input("Correo Electrónico Corporativo")
        cargo = st.selectbox("Cargo", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Constructor/Residente", "Estudiante", "Otro"])
        
        # Checkbox de privacidad (Importante para GDPR/Leyes de datos)
        acepto = st.checkbox("Acepto recibir información técnica relacionada.")
        
        submit = st.form_submit_button("🚀 INGRESAR AL SISTEMA")
        
        if submit:
            # Validaciones simples
            if not nombre or not email:
                st.error("Por favor ingrese al menos su Nombre y Correo.")
            elif "@" not in email or "." not in email:
                st.error("El correo electrónico no parece válido.")
            elif not acepto:
                st.warning("Debe aceptar los términos para continuar.")
            else:
                # --- AQUÍ GUARDARÍAS LOS DATOS ---
                # En una app real, aquí enviarías los datos a una base de datos (Google Sheets, Firebase, SQL)
                # Por ahora, solo simulamos el éxito.
                st.session_state['usuario_registrado'] = True
                st.session_state['datos_usuario'] = {'nombre': nombre, 'email': email}
                st.success(f"¡Bienvenido, {nombre}! Cargando sistema...")
                st.rerun() # Recargar la página para mostrar la app
# ==============================================================================
# 4. APLICACIÓN PRINCIPAL (EL PREMIO)
# ==============================================================================
def app_principal():
    # Barra lateral con info de sesión
    with st.sidebar:
        st.info(f"👤 Usuario: **{st.session_state['usuario_nombre']}**")
        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False
            st.rerun()
        st.markdown("---")
# ==============================================================================
# 1. APLICACIÓN PRINCIPAL (TU CÓDIGO ORIGINAL ENVUELTO)
# ==============================================================================
def app_principal():
    # Mensaje de bienvenida personalizado en la barra lateral
    with st.sidebar:
        st.success(f"Sesión activa: **{st.session_state['datos_usuario']['nombre']}**")
        if st.button("Cerrar Sesión"):
            st.session_state['usuario_registrado'] = False
            st.rerun()
        st.markdown("---")



# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Diseño Avanzado de Micropilotes", layout="wide", page_icon="🏗️")

# Estilos CSS para mejorar la visualización de las pestañas
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Sistema de Diseño de Micropilotes")
st.markdown("Optimización de diseño y análisis geotécnico integrado.")

# Crear pestañas principales
tab_diseno, tab_geo = st.tabs(["📐 Diseño & Optimización", "🌍 Correlaciones Geotécnicas (SPT)"])

# ==============================================================================
# PARÁMETROS FIJOS GLOBALES
# ==============================================================================
# Diámetros comerciales y su eficiencia (Velocidad)
DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85}
LISTA_D = sorted(list(DIAMETROS_COM.keys()))

# Restricciones de búsqueda
MIN_MICROS = 3
MAX_MICROS = 15
RANGO_L = range(5, 36) # Buscamos de 5m hasta 35m
COSTO_PERF_BASE = 100

# Factores Ambientales
FACTOR_CO2_CEMENTO = 0.90 # kg CO2e / kg
FACTOR_CO2_PERF = 15.0    # kg CO2e / m
FACTOR_CO2_ACERO = 1.85   # kg CO2e / kg
DENSIDAD_ACERO = 7850.0   # kg/m3
DENSIDAD_CEMENTO = 3150.0 # kg/m3

# Resistencia del Acero (Para cálculo simplificado de área)
FY_ACERO_KPA = 500000.0 # 500 MPa (Ej. Acero N80 o barras de alta resistencia)

# Colores para gráficos
COLORES_ESTRATOS = ["#D7BDE2", "#A9CCE3", "#A3E4D7", "#F9E79F", "#F5B7B1", "#D2B4DE", "#AED6F1", "#A2D9CE", "#F7DC6F", "#F1948A"]

# ==============================================================================
# ---------------------- PESTAÑA 1: DISEÑO & OPTIMIZACIÓN ----------------------
# ==============================================================================
with tab_diseno:
    # ==========================================
    # 1. BARRA LATERAL (INPUTS DE DISEÑO)
    # ==========================================
    with st.sidebar:
        st.header("1. Definición de Estratigrafía")
        
        num_capas = st.slider("Número de Capas de Suelo", min_value=1, max_value=10, value=3)
        
        ESTRATOS = []
        z_acumulada = 0.0
        
        with st.expander("Configurar Capas (Espesor, Adherencia, Expansión)", expanded=True):
            for i in range(num_capas):
                st.markdown(f"**--- Capa {i+1} ---**")
                c1, c2, c3 = st.columns(3)
                espesor = c1.number_input(f"Espesor C{i+1} (m)", value=3.0 if i==0 else 5.0, step=0.5, key=f"esp_{i}")
                qs = c2.number_input(f"Adherencia qs C{i+1} (kPa)", value=40.0 + i*20.0, step=5.0, key=f"qs_{i}")
                f_exp = c3.number_input(f"Factor Exp. C{i+1}", value=1.1 + i*0.05, step=0.1, min_value=1.0, max_value=3.0, key=f"fexp_{i}", help="Factor de sobreconsumo de grout en esta capa")
                
                z_acumulada += espesor
                ESTRATOS.append({
                    "espesor": espesor,
                    "z_fin": z_acumulada,
                    "qs": qs,
                    "f_exp": f_exp,
                    "color": COLORES_ESTRATOS[i % len(COLORES_ESTRATOS)]
                })

        st.caption(f"Profundidad total definida: {z_acumulada:.1f} m")

        st.header("2. Cargas y Seguridad")
        CARGA_TON = st.number_input("Carga Total en Cabezal (Ton)", value=120.0, step=1.0)
        FS_REQ = st.slider("Factor de Seguridad (Geotécnico)", 1.5, 3.0, 2.0, 0.1)
        
        st.header("3. Materiales (Lechada)")
        RELACION_AGUA_CEMENTO = st.slider("Relación A/C Lechada", 0.4, 0.6, 0.50, 0.05)
        st.caption("Nota: El acero se calcula automáticamente asumiendo Fy=500MPa para soportar la carga actuante.")

        st.divider()
        calcular = st.button("🚀 CALCULAR DISEÑO OPTIMIZADO", type="primary")

    # ==========================================
    # 2. FUNCIONES DE CÁLCULO (DISEÑO)
    # ==========================================
    def calc_capacidad_individual_dinamica(D, L_total, estratos_data):
        """Calcula Q_ult integrando la fricción en los estratos definidos."""
        Q_ult = 0
        z_actual = 0
        
        for estrato in estratos_data:
            if z_actual >= L_total: break
            
            # El techo del estrato actual es z_actual
            # El piso del estrato actual es estrato["z_fin"]
            
            # La profundidad efectiva en este estrato es el mínimo entre el piso del estrato y el largo total del micropilote
            z_fondo_efectivo = min(estrato["z_fin"], L_total)
            
            # Espesor de micropilote DENTRO de este estrato
            espesor_en_estrato = z_fondo_efectivo - z_actual
            
            if espesor_en_estrato > 0:
                area_lateral = math.pi * D * espesor_en_estrato
                Q_ult += area_lateral * estrato["qs"]
                
            z_actual = z_fondo_efectivo
            
        return Q_ult

    def calc_volumenes_grout(D, L_total, estratos_data):
        """Calcula volumen teórico y expandido basado en los factores de cada capa."""
        vol_teo_total = 0
        vol_exp_total = 0
        z_actual = 0
        area_perf = math.pi * (D/2)**2

        for estrato in estratos_data:
            if z_actual >= L_total: break
            
            z_fondo_efectivo = min(estrato["z_fin"], L_total)
            espesor_en_estrato = z_fondo_efectivo - z_actual
            
            if espesor_en_estrato > 0:
                v_teo_capa = area_perf * espesor_en_estrato
                v_exp_capa = v_teo_capa * estrato["f_exp"]
                
                vol_teo_total += v_teo_capa
                vol_exp_total += v_exp_capa
            
            z_actual = z_fondo_efectivo
            
        return vol_teo_total, vol_exp_total

    def get_peso_cemento_m3(wc):
        # kg cemento por m3 de lechada
        return 1.0 / (wc/1000.0 + 1.0/DENSIDAD_CEMENTO)

    def calc_impacto_ambiental_simplificado(L_total, N_micros, vol_grout_exp_total, Q_act_kpa_por_pilote):
        """
        Calcula CO2. El acero se calcula para resistir Q_act con Fy=500MPa.
        """
        # 1. Acero (Estructural)
        # Área necesaria = Carga Actuante / Fy
        area_acero_necesaria_m2 = Q_act_kpa_por_pilote / FY_ACERO_KPA
        vol_acero_total = area_acero_necesaria_m2 * L_total * N_micros
        peso_acero_total = vol_acero_total * DENSIDAD_ACERO
        
        # 2. Grout
        # Asumimos que el volumen expandido calculado ya considera el llenado total.
        # Restamos el volumen que ocupa el acero para no duplicar, aunque el factor de expansión suele cubrir esto.
        # Para ser conservadores en CO2, usemos el vol_grout_exp_total directo para el cemento, 
        # o restemos el acero si queremos ser más precisos en el volumen neto de mezcla.
        # Vamos a restar el acero para ser consistentes físicamente.
        vol_grout_neto = max(0, vol_grout_exp_total - vol_acero_total)
        peso_cemento_total = vol_grout_neto * get_peso_cemento_m3(RELACION_AGUA_CEMENTO)
        
        # 3. Perforación
        metros_perf_totales = L_total * N_micros
        
        # Cálculo total CO2
        co2_total = (peso_acero_total*FACTOR_CO2_ACERO + peso_cemento_total*FACTOR_CO2_CEMENTO + metros_perf_totales*FACTOR_CO2_PERF) / 1000
        return co2_total, area_acero_necesaria_m2 * 10000 # Devolver también cm2 de acero para referencia

    def obtener_perfil_resistencia_grafico(D, L, estratos_data):
        """Genera datos para graficar la curva de transferencia."""
        z_points = [0]
        q_points = [0]
        z_actual = 0
        q_acum = 0
        
        for estrato in estratos_data:
            if z_actual >= L: break
            z_fondo_tramo = min(estrato["z_fin"], L)
            espesor = z_fondo_tramo - z_actual
            if espesor > 0:
                q_tramo = (math.pi * D * espesor) * estrato["qs"]
                q_acum += q_tramo
                z_points.append(z_fondo_tramo)
                q_points.append(q_acum)
            z_actual = z_fondo_tramo
        return z_points, q_points

    # ==========================================
    # 3. LÓGICA PRINCIPAL (DISEÑO)
    # ==========================================
    if calcular:
        CARGA_REQ_KN = CARGA_TON * 9.81
        resultados_raw = []

        with st.spinner('Optimizando diseño según estratigrafía definida...'):
            for D in LISTA_D:
                # Ya no validamos recubrimiento aquí porque el acero no es input fijo.

                for N in range(MIN_MICROS, MAX_MICROS + 1):
                    Q_act_por_pilote_kn = CARGA_REQ_KN / N
                    Q_req_geotec_por_pilote = Q_act_por_pilote_kn * FS_REQ
                    
                    for L in RANGO_L:
                        # 1. Capacidad Geotécnica Última
                        Q_ult_geo = calc_capacidad_individual_dinamica(D, L, ESTRATOS)
                        
                        # Verificar si cumple geotecnia
                        if Q_ult_geo >= Q_req_geotec_por_pilote:
                            FS_calc = Q_ult_geo / Q_act_por_pilote_kn
                            
                            # 2. Cálculos Volumétricos (con expansión variable)
                            v_teo_mic, v_exp_mic = calc_volumenes_grout(D, L, ESTRATOS)
                            v_teo_total = v_teo_mic * N
                            v_exp_total = v_exp_mic * N
                            
                            # 3. Cálculos Económicos
                            eficiencia = DIAMETROS_COM[D]
                            costo_idx = (L * N * COSTO_PERF_BASE) / eficiencia
                            
                            # 4. Cálculo Ambiental (Acero simplificado)
                            # Necesitamos la carga actuante en kPa para la función de acero
                            Q_act_por_pilote_kpa = Q_act_por_pilote_kn # kN, y Fy está en kPa, la division da m2. Correcto.
                            co2, area_acero_cm2 = calc_impacto_ambiental_simplificado(L, N, v_exp_total, Q_act_por_pilote_kpa)
                            
                            resultados_raw.append({
                                "D_val": D, "D_mm": int(D*1000),
                                "N": N, "L_m": L, "L_Tot_m": L * N,
                                "FS": FS_calc, 
                                "Vol_Teo": v_teo_total, "Vol_Exp": v_exp_total,
                                "Costo_Idx": costo_idx, "Eficiencia": eficiencia,
                                "CO2_ton": co2,
                                "Q_adm_geo": Q_ult_geo / FS_REQ / 9.81, # Ton
                                "Q_act": Q_act_por_pilote_kn / 9.81,    # Ton
                                "Acero_req_cm2": area_acero_cm2 # Dato informativo
                            })
                            break # Encontramos L mínima para este N y D

        # --- VISUALIZACIÓN RESULTADOS DISEÑO ---
        if not resultados_raw:
            st.error("No se encontraron soluciones. Intente aumentar la profundidad de los estratos, mejorar la adherencia o aumentar la cantidad máxima de micropilotes.")
        else:
            # Selección Estratégica Top 10
            resultados_raw.sort(key=lambda x: x["Costo_Idx"])
            top_10 = []
            ids = set()
            
            # Asegurar variedad de diámetros
            for d_target in LISTA_D:
                cand = next((r for r in resultados_raw if r["D_val"] == d_target), None)
                if cand:
                    uid = f"{cand['D_mm']}-{cand['N']}-{cand['L_m']}"
                    if uid not in ids: top_10.append(cand); ids.add(uid)
            
            # Rellenar
            for r in resultados_raw:
                if len(top_10) >= 12: break # Mostremos hasta 12
                uid = f"{r['D_mm']}-{r['N']}-{r['L_m']}"
                if uid not in ids: top_10.append(r); ids.add(uid)

            top_10.sort(key=lambda x: x["Costo_Idx"])
            df = pd.DataFrame(top_10)

            # --- DASHBOARD ---
            best = df.iloc[0]
            
            # KPIs
            st.subheader("🏆 Mejor Opción (Balance Costo/Eficiencia)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Configuración", f"{int(best['N'])} micros x Ø{int(best['D_mm'])}mm")
            k2.metric("Longitud Total", f"{int(best['L_m'])} m/micro", f"Perf. Total: {best['L_Tot_m']}m")
            k3.metric("Volumen Grout (Expandido)", f"{best['Vol_Exp']:.1f} m³", help="Suma de volúmenes considerando el factor de expansión de cada estrato.")
            k4.metric("Huella CO2 Est.", f"{best['CO2_ton']:.1f} Ton", help="Basado en acero requerido para la carga (Fy=500MPa) y consumo de grout/diésel.")
            
            st.caption(f"Acero estructural requerido aprox. por micropilote: {best['Acero_req_cm2']:.1f} cm²")

            st.divider()
            
            col_graf, col_tabla = st.columns([1, 1.2])
            
            with col_graf:
                st.subheader("📉 Transferencia de Carga")
                fig_prof, ax_prof = plt.subplots(figsize=(8, 6))
                
                # Fondo estratos
                y_lim_plot = max([r['L_m'] for r in top_10]) + 2
                prev_z = 0
                for e in ESTRATOS:
                    h_layer = min(e["z_fin"], y_lim_plot) - prev_z
                    if h_layer > 0:
                        rect = patches.Rectangle((0, prev_z), 2000, h_layer, color=e["color"], alpha=0.3)
                        ax_prof.add_patch(rect)
                        ax_prof.text(5, prev_z + h_layer/2, f"qs={int(e['qs'])} | F.Exp={e['f_exp']}", color='#555555', va='center', fontsize=8)
                    prev_z = e["z_fin"]
                    if prev_z >= y_lim_plot: break

                # Curvas
                diametros_vistos = set()
                plotted_count = 0
                for i, row in df.iterrows():
                    if row['D_mm'] in diametros_vistos and plotted_count > 2: continue
                    
                    z_p, q_p = obtener_perfil_resistencia_grafico(row['D_val'], row['L_m'], ESTRATOS)
                    q_p_adm_ton = [(q / FS_REQ) / 9.81 for q in q_p]
                    
                    label_str = f"Ø{row['D_mm']} (Qadm: {q_p_adm_ton[-1]:.0f}T)"
                    ax_prof.plot(q_p_adm_ton, z_p, marker='o', markersize=4, linewidth=2, label=label_str, alpha=0.8 if i==0 else 0.5)
                    diametros_vistos.add(row['D_mm'])
                    plotted_count +=1

                ax_prof.set_ylim(y_lim_plot, 0)
                ax_prof.set_xlim(0, 90)
                ax_prof.set_ylabel("Profundidad (m)")
                ax_prof.set_xlabel("Capacidad Admisible Acumulada (Ton)")
                ax_prof.legend(loc='lower right', fontsize=8)
                ax_prof.grid(True, linestyle=':', alpha=0.5)
                st.pyplot(fig_prof)

            with col_tabla:
                st.subheader("📋 Mejores Alternativas")
                df_show = df[["D_mm", "N", "L_m", "L_Tot_m", "FS", "Q_adm_geo", "Q_act", "Vol_Exp", "CO2_ton"]].copy()
                df_show.columns = ["Ø(mm)", "Cant", "L(m)", "Perf(m)", "FS", "Qadm(T)", "Qact(T)", "Grout(m³)", "CO2(T)"]
                
                st.dataframe(
                    df_show.style.background_gradient(subset=["Perf(m)"], cmap="Blues_r")
                           .background_gradient(subset=["CO2(T)"], cmap="YlGn_r")
                           .format("{:.1f}", subset=["Qadm(T)", "Qact(T)", "Grout(m³)", "CO2(T)"])
                           .format("{:.2f}", subset=["FS"]),
                    use_container_width=True,
                    height=400
                )
    else:
        st.info("👈 Configure la estratigrafía y cargas en la barra lateral, luego presione 'CALCULAR'.")

# ==============================================================================
# ---------------------- PESTAÑA 2: CORRELACIONES GEOTÉCNICAS ------------------
# ==============================================================================
with tab_geo:
    st.header("🌍 Estimación de Adherencia ($q_s$) mediante SPT")
    
    col_input, col_ref = st.columns([1, 1.5])
    
    with col_input:
        st.subheader("1. Ingreso de Datos SPT")
        st.markdown("Pegue sus datos de sondeo (Profundidad, N_SPT) separados por coma o tabulación.")
        
        spt_data_raw = st.text_area("Datos (Profundidad [m], N_golpes)", height=300,
                                    value="1.5, 4\n3.0, 7\n4.5, 12\n6.0, 15\n7.5, 22\n9.0, 28\n10.5, 35\n12.0, 42\n15.0, 50")
        
        st.subheader("2. Factor de Correlación (K)")
        st.markdown(r"Modelo simplificado: $q_s (kPa) \approx K \cdot N_{SPT}$")
        k_factor = st.slider("Factor K", min_value=1.0, max_value=10.0, value=3.5, step=0.5, 
                             help="Valor típico entre 2 y 5 para inyección tipo A/B. Valores mayores para inyección repetida (tipo D).")
        
        process_spt = st.button("Procesar y Graficar")

    with col_ref:
        st.subheader("Referencia FHWA (NHI-05-039)")
        st.markdown("""
        Valores típicos de adherencia última ($q_s$) para micropilotes inyectados a gravedad (Tipo A):
        
        | Tipo de Suelo | Adherencia $q_s$ (kPa) |
        | :--- | :--- |
        | Arcilla Blanda / Limo | 20 - 60 |
        | Arcilla Media / Limo Arenoso | 40 - 90 |
        | Arcilla Dura / Arena Fina Densa | 80 - 150 |
        | Arena Media-Densa / Grava | 100 - 250 |
        | Roca Meteorizada / Esquisto | 200 - 500+ |
        
        *Nota: El uso de SPT es una aproximación grosera. Se recomienda usar datos de laboratorio (Su, phi) o pruebas de carga para diseños finales.*
        """)

    st.divider()

    if process_spt and spt_data_raw:
        try:
            # Procesar datos
            data_io = StringIO(spt_data_raw)
            df_spt = pd.read_csv(data_io, names=["Profundidad", "N_SPT"], header=None, sep=r'[,\t]+', engine='python')
            
            # Calcular qs estimado
            df_spt["qs_est"] = df_spt["N_SPT"] * k_factor
            
            # Graficar
            st.subheader("Resultados de Correlación")
            fig_spt, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            
            # Gráfica N-SPT
            ax1.plot(df_spt["N_SPT"], df_spt["Profundidad"], marker='o', linestyle='-', color='blue', label='N-SPT')
            ax1.set_ylim(df_spt["Profundidad"].max() + 2, 0)
            ax1.set_title("Perfil N-SPT")
            ax1.set_xlabel("N Golpes")
            ax1.set_ylabel("Profundidad (m)")
            ax1.grid(True, linestyle=':')
            
            # Gráfica qs estimado
            ax2.plot(df_spt["qs_est"], df_spt["Profundidad"], marker='s', linestyle='-', color='red', label=f'qs = {k_factor}*N')
            ax2.fill_betweenx(df_spt["Profundidad"], 0, df_spt["qs_est"], color='red', alpha=0.2)
            ax2.set_title(f"Adherencia Estimada (K={k_factor})")
            ax2.set_xlabel("qs Estimado (kPa)")
            ax2.grid(True, linestyle=':')
            ax2.legend()
            
            st.pyplot(fig_spt)
            
            st.markdown("✅ **Listo.** Use los valores de la gráfica roja para alimentar la pestaña de 'Diseño & Optimización'.")
            
        except Exception as e:
            st.error(f"Error al procesar los datos. Asegúrese del formato:\n\n`Profundidad, N_valor`\n\nError detallado: {e}")

# 5. CONTROL DE FLUJO
# ==============================================================================
if 'usuario_registrado' not in st.session_state:
    st.session_state['usuario_registrado'] = False

if st.session_state['usuario_registrado']:
    app_principal()
else:
    mostrar_registro()

