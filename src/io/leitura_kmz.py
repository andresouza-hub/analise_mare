# =========================================================
# LEITURA DE KMZ (POÇOS) + CONVERSÃO PARA UTM
# =========================================================
# - Extrai poços (Placemark/Point)
# - Converte lon/lat -> UTM (m)
# - NÃO usa cota do KMZ
# =========================================================

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re

from pyproj import CRS, Transformer


def _normalizar_nome_poco(nome: str) -> str:
    nome = str(nome).strip().upper()
    nome = re.sub(r'\s+', ' ', nome)
    return nome


def _extrair_kml_do_kmz(caminho_kmz: str) -> bytes:
    with zipfile.ZipFile(caminho_kmz, 'r') as z:
        kmls = [n for n in z.namelist() if n.lower().endswith('.kml')]
        if not kmls:
            raise ValueError("KMZ não contém arquivo .kml.")
        return z.read(kmls[0])


def _parse_kml_pocos(kml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(kml_bytes)

    def local(tag):
        return tag.split('}', 1)[-1]

    rows = []

    for pm in root.iter():
        if local(pm.tag) != 'Placemark':
            continue

        nome = None
        coord = None

        for ch in pm.iter():
            if local(ch.tag) == 'name' and ch.text:
                nome = ch.text.strip()
            if local(ch.tag) == 'coordinates' and ch.text:
                coord = ch.text.strip()

        if not nome or not coord:
            continue

        first = coord.split()[0]
        parts = first.split(',')
        if len(parts) < 2:
            continue

        lon = float(parts[0])
        lat = float(parts[1])

        rows.append({
            'Poco': _normalizar_nome_poco(nome),
            'lon': lon,
            'lat': lat
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=['Poco']).reset_index(drop=True)

    if df.empty:
        raise ValueError("Nenhum poço (Placemark/Point) foi encontrado no KMZ.")

    return df


def _converter_lonlat_para_utm(df: pd.DataFrame) -> pd.DataFrame:
    lon0 = float(df['lon'].mean())
    lat0 = float(df['lat'].mean())

    zona = int((lon0 + 180) / 6) + 1
    sul = lat0 < 0

    epsg = 32700 + zona if sul else 32600 + zona

    crs_src = CRS.from_epsg(4326)
    crs_dst = CRS.from_epsg(epsg)
    tf = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    x, y = tf.transform(df['lon'].to_numpy(), df['lat'].to_numpy())

    out = df.copy()
    out['X_m'] = x
    out['Y_m'] = y

    return out[['Poco', 'X_m', 'Y_m']]


def ler_kmz_pocos(caminho_kmz: str) -> pd.DataFrame:
    """
    Lê arquivo KMZ e retorna DataFrame com:
      - Poco
      - X_m
      - Y_m
    """
    kml_bytes = _extrair_kml_do_kmz(caminho_kmz)
    df_lonlat = _parse_kml_pocos(kml_bytes)
    df_utm = _converter_lonlat_para_utm(df_lonlat)
    return df_utm
