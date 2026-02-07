import os
import tempfile
import pandas as pd
import streamlit as st

from src.core.orquestrador_b import executar_analise_b
from src.viz.mapas_gradiente import salvar_mapa_vetor, angulo_dominante_por_hora


st.set_page_config(page_title="Agente GAC | Maré, Lag e Gradiente", layout="wide")

st.title("Agente GAC – Maré/Lag + Gradiente 2D")

st.write("Carregue os arquivos dos poços, defina o início por poço e (opcionalmente) maré/cotas/kmz.")

OUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Upload poços
# ---------------------------
pocos_files = st.file_uploader(
    "Arquivos dos poços (Level Logger) – selecione 1 ou mais",
    type=None,
    accept_multiple_files=True
)

inicio_por_poco = {}

caminhos_temp_pocos = []
if pocos_files:
    st.subheader("Início do teste por poço")
    for f in pocos_files:
        nome = os.path.splitext(f.name)[0].strip()
        s = st.text_input(f"Início para {nome} (dd/mm/aaaa hh:mm)", value="")
        if s.strip():
            try:
                inicio_por_poco[nome] = pd.to_datetime(s, dayfirst=True)
            except Exception:
                st.error(f"Formato inválido em '{nome}'. Use dd/mm/aaaa hh:mm")

        # salvar temporário
        tmp_path = os.path.join(tempfile.gettempdir(), f.name)
        with open(tmp_path, "wb") as w:
            w.write(f.getbuffer())
        caminhos_temp_pocos.append(tmp_path)

# ---------------------------
# Upload maré/cotas/kmz
# ---------------------------
st.subheader("Arquivos opcionais")
mare_file = st.file_uploader("Maré (CSV de eventos com coluna 'Ano')", type=["csv"])
cotas_file = st.file_uploader("Cotas (Excel com Poco, Cota_TOC_m)", type=["xlsx"])
kmz_file = st.file_uploader("KMZ (poços com coordenadas)", type=["kmz"])

def _save_tmp(upload, suffix):
    if upload is None:
        return None
    tmp_path = os.path.join(tempfile.gettempdir(), upload.name)
    with open(tmp_path, "wb") as w:
        w.write(upload.getbuffer())
    return tmp_path

caminho_mare = _save_tmp(mare_file, ".csv")
caminho_cotas = _save_tmp(cotas_file, ".xlsx")
caminho_kmz = _save_tmp(kmz_file, ".kmz")

# ---------------------------
# Executar
# ---------------------------
if st.button("Executar análise"):
    if not pocos_files:
        st.error("Você precisa carregar pelo menos 1 arquivo de poço.")
        st.stop()

    # checar inícios
    nomes = [os.path.splitext(f.name)[0].strip() for f in pocos_files]
    faltando = [n for n in nomes if n not in inicio_por_poco]
    if faltando:
        st.error(f"Faltou definir início para: {faltando}")
        st.stop()

    with st.spinner("Rodando..."):
        out = executar_analise_b(
            caminhos_pocos=caminhos_temp_pocos,
            inicio_por_poco=inicio_por_poco,
            caminho_mare=caminho_mare,
            caminho_cotas=caminho_cotas,
            caminho_kmz=caminho_kmz,
        )

    st.success("Concluído.")

    # ---------------------------
    # Mostrar consolidado
    # ---------------------------
    st.subheader("Consolidado")
    st.dataframe(out["consolidado"])

    # ---------------------------
    # Exportar Excel consolidado
    # ---------------------------
    xlsx_path = os.path.join(OUT_DIR, "resultado_consolidado.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        out["consolidado"].to_excel(writer, sheet_name="Consolidado", index=False)
        for poco, dfp in out["por_poco"].items():
            sheet = poco[:31]
            dfp.to_excel(writer, sheet_name=sheet, index=False)

        if out["gradiente_2d"] and out["gradiente_2d"]["horario"] is not None:
            out["gradiente_2d"]["horario"].to_excel(writer, sheet_name="Gradiente2D_Horario", index=False)

    st.info(f"Arquivo salvo: {xlsx_path}")

    # ---------------------------
    # Gradiente 2D + mapas
    # ---------------------------
    g2d = out.get("gradiente_2d", None)
    if g2d is None:
        st.warning("Gradiente 2D não foi executado (faltou cotas e/ou kmz).")
        st.stop()

    if g2d["horario"] is None or g2d["horario"].empty:
        st.warning("Gradiente 2D horário não gerou vetores suficientes.")
    else:
        df_grad_h = g2d["horario"]
        st.subheader("Gradiente 2D horário")
        st.dataframe(df_grad_h)

        csv_path = os.path.join(OUT_DIR, "gradiente_2d_horario.csv")
        df_grad_h.to_csv(csv_path, index=False)
        st.info(f"CSV salvo: {csv_path}")

    # Mapas (se tiver insumos OK)
    ins = out.get("insumos_gradiente", None)
    if ins and ins.get("status") == "OK":
        df_xy = ins["coords_xy"]

        # SERFES baseline (se OK)
        serfes = g2d.get("serfes", {})
        if serfes and serfes.get("status") == "OK":
            p = os.path.join(OUT_DIR, "Mapa_SERFES.png")
            salvar_mapa_vetor(
                df_xy=df_xy,
                angulo_fluxo_deg=serfes["angulo_fluxo_deg"],
                out_png=p,
                titulo=f"Gradiente 2D (Serfes) | {serfes['angulo_fluxo_deg']:.1f}°"
            )
            st.image(p)

        # Horário dominante / inversão
        if g2d.get("horario") is not None and not g2d["horario"].empty:
            inv = g2d.get("inversao", {})
            if inv.get("inversao") is False:
                ang_dom = angulo_dominante_por_hora(g2d["horario"])
                p = os.path.join(OUT_DIR, "Mapa_HORARIO_Dominante.png")
                salvar_mapa_vetor(df_xy, ang_dom, p, titulo=f"Gradiente 2D Horário | Dominante {ang_dom:.1f}°")
                st.image(p)
            else:
                st.warning(f"Inversão detectada: {inv}")
                # Aqui, por enquanto, mostramos só um aviso e os dados.
                # Se você quiser, na próxima etapa eu separo automaticamente PRE/PÓS e gero 2 mapas.
    else:
        st.warning("Sem coords_xy/cotas OK para mapear.")
