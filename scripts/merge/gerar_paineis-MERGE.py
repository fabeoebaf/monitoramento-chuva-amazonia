import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from datetime import datetime

try:
    from adjustText import adjust_text
except ImportError:
    print("ERRO: Instale 'adjustText' rodando 'pip install adjustText'")
    exit()

# =============================================================================
# --- 1. CONFIGURAÇÕES DE DIRETÓRIOS ---
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DIR_OBSERVADO  = os.path.join(BASE_DIR, "data", "merge", "observado")
DIR_ANOMALIA   = os.path.join(BASE_DIR, "data", "merge", "anomalia")

DIR_SAIDA_COMP = os.path.join(BASE_DIR, "outputs", "figuras_comparativo")
DIR_SAIDA_ANOM = os.path.join(BASE_DIR, "outputs", "figuras_anomalia")

SHP_BACIAS       = os.path.join(BASE_DIR, "data", "shapes", "shp_bacias", "BACIA_AMAZONICA_.shp")
SHP_RIOS         = os.path.join(BASE_DIR, "data", "shapes", "shp_rios", "geometrias-corrigidas-Massa-Agua.shp")
SHP_SUB_SOLIMOES = os.path.join(BASE_DIR, "data", "shapes", "shp_sub_solimoes", "sub_solimoes.shp")
SHP_AMAZONAS     = os.path.join(BASE_DIR, "data", "shapes", "amazonas", "AM_UF_2024.shp")
CAMINHO_DEM      = os.path.join(BASE_DIR, "data", "shapes", "mde", "bacia.asc")

# =============================================================================
# --- 2. DADOS AUXILIARES E LIMITES VISUAIS ---
# =============================================================================

LIMS_MENSAL = {"vmin_anom": -200, "vmax_anom": 200, "vmax_obs": 500}
LIMS_TRIMES = {"vmin_anom": -450, "vmax_anom": 450, "vmax_obs": 1500}

NOME_MESES = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
              7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"}

SUB_BACIAS = {"48": "Rio Negro", "46": "Rio Madeira", "49": "Rio Solimoes"}

DADOS_CIDADES = [
    {"nome": "Tabatinga",     "lat": -4.2347,  "lon": -69.9447},
    {"nome": "Coari",         "lat": -4.0856,  "lon": -63.0833},
    {"nome": "Manacapuru",    "lat": -3.3106,  "lon": -60.6094},
    {"nome": "Beruri",        "lat": -3.8989,  "lon": -61.3742},
    {"nome": "Lábrea",        "lat": -7.2581,  "lon": -64.7975},
    {"nome": "Rio Branco",    "lat": -9.9758,  "lon": -67.8000},
    {"nome": "Barcelos",      "lat": -0.9658,  "lon": -62.9311},
    {"nome": "Moura",         "lat": -1.4567,  "lon": -61.6347},
    {"nome": "Manaus",        "lat": -3.1383,  "lon": -60.0272},
    {"nome": "Porto Velho",   "lat": -8.7483,  "lon": -63.9169},
    {"nome": "Humaitá",       "lat": -7.5028,  "lon": -63.0183},
    {"nome": "Manicoré",      "lat": -5.8167,  "lon": -61.3019},
    {"nome": "Ji-Paraná",     "lat": -10.8736, "lon": -61.9356},
    {"nome": "Itacoatiara",   "lat": -3.1539,  "lon": -58.4114},
    {"nome": "Óbidos",        "lat": -1.9192,  "lon": -55.5131},
    {"nome": "Eirunepé",                 "lat": -6.684444,  "lon": -69.8811},
    {"nome": "Ipixuna",                  "lat": -7.050833,  "lon": -71.68417},
    {"nome": "Cruzeiro do Sul",             "lat": -7.610556,  "lon": -72.68111},
    {"nome": "Manoel Urbano",               "lat": -8.884167,  "lon": -69.26806},
    {"nome": "Boca do Acre",                "lat": -8.735556,  "lon": -67.4000},
    {"nome": "Palmeiras do Javari",         "lat": -5.133333,  "lon": -72.8000},
    {"nome": "Fonte Boa (Solimões)",        "lat": -2.491389,  "lon": -66.06167},
    {"nome": "Tefé",                        "lat": -3.375833,  "lon": -64.65472},
    {"nome": "São Gabriel da Cachoeira", "lat": -0.136111,  "lon": -67.08472},
    {"nome": "Boa Vista",                   "lat":  2.826111,  "lon": -60.65805},
    {"nome": "Caracaraí",                   "lat":  1.821389,  "lon": -61.12361},
    {"nome": "Príncipe da Beira",           "lat": -12.42667,  "lon": -64.42528},
    {"nome": "Abunã",                       "lat": -9.7053,    "lon": -65.3667},
    {"nome": "Guajará-Mirim",               "lat": -10.7925,   "lon": -65.3478},
]

DF_CIDADES = pd.DataFrame(DADOS_CIDADES)

# =============================================================================
# --- 3. FUNÇÕES DE PROCESSAMENTO ---
# =============================================================================

def encontrar_arquivo(diretorio, ano, mes, tipo):
    pasta = os.path.join(diretorio, str(ano), f"{mes:02d}")
    if os.path.exists(pasta):
        arquivos = glob.glob(os.path.join(pasta, "*.nc"))
        mensais = [f for f in arquivos if "mensal" in f.lower()]
        if mensais: return mensais[0]
        if tipo == "anomalia":
            acumulados = [f for f in arquivos if "acumulada" in f.lower()]
            if acumulados: return sorted(acumulados, reverse=True)[0]
    if tipo == "observado":
        arq_anual = os.path.join(diretorio, f"MERGE_GPM_COMPLETO_{ano}.nc")
        if os.path.exists(arq_anual): return arq_anual
    return None

def carregar_dados_base(nc_path, ano, mes, eh_observado_bruto=False):
    if not nc_path: return None
    try:
        ds = xr.open_dataset(nc_path)
        vars_validas = [v for v in ds.data_vars if v.lower() not in ['crs', 'spatial_ref', 'time_bnds'] and ds[v].ndim >= 2]
        da = ds[vars_validas[0]].copy()
        ds.close()

        rename = {c: 'x' for c in da.coords if c.lower() in ['longitude', 'lon', 'x']}
        rename.update({c: 'y' for c in da.coords if c.lower() in ['latitude', 'lat', 'y']})
        if rename: da = da.rename(rename)
        
        if 'x' in da.coords and da.x.max() > 180: da = da.assign_coords(x=(((da.x + 180) % 360) - 180))
        if 'x' in da.coords and 'y' in da.coords: da = da.sortby(['x', 'y'])

        if eh_observado_bruto and 'time' in da.dims:
            try: da = da.sel(time=f"{ano}-{mes:02d}").sum(dim='time', skipna=False)
            except: pass
        else: da = da.squeeze()
        
        # 🔥 A MÁGICA AQUI: Retorna UM ARRAY PURO, sem nenhum metadado ou CRS.
        # Assim, as somas trimestrais não vão corromper o arquivo.
        da_pure = xr.DataArray(
            da.values, 
            coords={"y": da.y.values, "x": da.x.values}, 
            dims=("y", "x")
        )
        return da_pure
        
    except Exception as e:
        print(f" ❌ Erro ao carregar base {os.path.basename(nc_path)}: {e}")
        return None

def obter_dados_trimestre(trimestre, ano_referencia, tipo):
    meses = [(ano_referencia - 1, 11), (ano_referencia - 1, 12), (ano_referencia, 1)] if trimestre == "NDJ" else [(ano_referencia - 1, 12), (ano_referencia, 1), (ano_referencia, 2)]
    da_acumulado = None
    dir_busca = DIR_OBSERVADO if tipo == "observado" else DIR_ANOMALIA
    
    for a, m in meses:
        path = encontrar_arquivo(dir_busca, a, m, tipo)
        if not path: return None
        da = carregar_dados_base(path, a, m, eh_observado_bruto=("COMPLETO" in path if tipo == "observado" else False))
        if da is None: return None
        da_acumulado = da if da_acumulado is None else da_acumulado + da
    return da_acumulado

def processar_recorte(da_base, gdf_recorte):
    try:
        # Garante a projeção limpa
        da_base = da_base.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("epsg:4326")
        
        if gdf_recorte.crs != "epsg:4326": 
            gdf_recorte = gdf_recorte.to_crs("epsg:4326")
            
        da_clipped = da_base.rio.clip(gdf_recorte.geometry.values, gdf_recorte.crs, drop=True)
        
        # Extrai os valores puros em formato Numpy (Isso evita o erro ndim=1)
        x_arr = da_clipped.x.values
        y_arr = da_clipped.y.values
        val_arr = da_clipped.values
        
        # Reconstrói limpo e preenche buracos com 0
        da_clean = xr.DataArray(val_arr, coords={"y": y_arr, "x": x_arr}, dims=("y", "x")).fillna(0)
        
        if da_clean.size > 0:
            # AGORA SIM! x_arr.min() é um float puro, a interpolação vai voar!
            fator = 4
            new_x = np.linspace(x_arr.min(), x_arr.max(), int(len(x_arr) * fator))
            new_y = np.linspace(y_arr.min(), y_arr.max(), int(len(y_arr) * fator))
            
            return da_clean.interp(x=new_x, y=new_y, method='cubic')
            
        return da_clipped
        
    except Exception as e:
        print(f"   [AVISO] Erro no processamento de recorte: {e}")
        # Mostra a linha exata caso aconteça de novo
        import traceback
        traceback.print_exc()
        return None

def calcular_hillshade(dem_path, gdf_recorte):
    if not os.path.exists(dem_path): return None, None
    rds = rioxarray.open_rasterio(dem_path)
    if rds.rio.crs != gdf_recorte.crs: rds = rds.rio.reproject(gdf_recorte.crs)
    rds_clip = rds.rio.clip(gdf_recorte.geometry.values, gdf_recorte.crs, drop=True)
    data = np.where(rds_clip.isel(band=0).values.astype('float32') < -500, np.nan, rds_clip.isel(band=0).values.astype('float32'))
    hs = LightSource(azdeg=315, altdeg=45).hillshade(data, vert_exag=2000)
    return hs, [rds_clip.rio.bounds()[i] for i in [0, 2, 1, 3]]

# =============================================================================
# --- 4. EXPORTAÇÃO CSV ---
# =============================================================================

def exportar_csv_cidades(da_obs, da_anom, df_cidades, nome_saida_csv):
    resultados = []
    for _, row in df_cidades.iterrows():
        lat, lon = row['lat'], row['lon']
        try:
            obs_val = float(da_obs.sel(x=lon, y=lat, method='nearest').values)
            anom_val = float(da_anom.sel(x=lon, y=lat, method='nearest').values)
        except:
            obs_val, anom_val = np.nan, np.nan
            
        resultados.append({
            "Cidade": row['nome'], "Latitude": lat, "Longitude": lon,
            "Precipitacao_MERGE_mm": round(obs_val, 1), "Anomalia_mm": round(anom_val, 1)
        })
        
    pd.DataFrame(resultados).to_csv(nome_saida_csv, index=False, sep=';', decimal=',')
    print(f"   -> Tabela exportada: {os.path.basename(nome_saida_csv)}")

# =============================================================================
# --- 5. PLOTAGEM ---
# =============================================================================

def plotar_mapa(tipo_mapa, da_obs, da_anom, gdf_recorte, gdf_rios, gdf_cid, titulo, nome_saida, dem_path, lims):
    hs_arr, hs_ext = calcular_hillshade(dem_path, gdf_recorte)
    
    vals_obs = da_obs.values
    mean_obs = float(np.nanmean(vals_obs[vals_obs > 0.1])) if np.any(vals_obs > 0.1) else 0.0
    mean_anom = float(da_anom.mean(skipna=True))
    
    mean_clim = mean_obs - mean_anom
    txt_perc = f"({'+' if (mean_anom/mean_clim)*100 > 0 else ''}{(mean_anom/mean_clim)*100:.1f}%)" if mean_clim > 1 else "(-)"
    
    umidade_relativa, probabilidade = "--", "--"
    
    txt_box_obs = f"Média Espacial: {mean_obs:.1f} mm\nMáximo: {float(np.nanmax(vals_obs)):.1f} mm"
    txt_box_anom = (f"Média: {mean_anom:+.1f} mm {'↑' if mean_anom >= 0 else '↓'}\n"
                    f"{txt_perc} da Normal\n"
                    f"UR Prevista: {umidade_relativa} %\n"
                    f"Probabilidade: {probabilidade} %")

    if tipo_mapa == "comparativo":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
        eixos = [ax1, ax2]
        
        if hs_arr is not None: ax1.imshow(hs_arr, cmap='gray', extent=hs_ext, origin='upper', alpha=1.0, zorder=1)
        im1 = da_obs.plot(ax=ax1, cmap='YlGnBu', vmin=0, vmax=lims["vmax_obs"], add_colorbar=False, alpha=0.9, zorder=2, rasterized=True)
        ax1.set_title(f"Chuva Observada - MERGE\n{titulo}", fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.03, aspect=40, extend='max')
        cbar1.set_label('Precipitação (mm)', fontsize=12)
        ax1.text(0.97, 0.97, txt_box_obs, transform=ax1.transAxes, fontsize=11, fontweight='bold', ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"), zorder=7)
        ax_anom = ax2
    else: 
        fig, ax_anom = plt.subplots(figsize=(14, 10), constrained_layout=True)
        eixos = [ax_anom]

    if hs_arr is not None: ax_anom.imshow(hs_arr, cmap='gray', extent=hs_ext, origin='upper', alpha=1.0, zorder=1)
    im2 = da_anom.plot(ax=ax_anom, cmap='BrBG', vmin=lims["vmin_anom"], vmax=lims["vmax_anom"], add_colorbar=False, alpha=0.9, zorder=2, rasterized=True)
    ax_anom.set_title(f"Anomalia de Chuva - MERGE\n{titulo}", fontsize=20 if tipo_mapa == "anomalia" else 14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax_anom, orientation='horizontal', pad=0.03, aspect=50 if tipo_mapa == "anomalia" else 40, extend='both')
    cbar2.set_label('Anomalia (mm)', fontsize=14 if tipo_mapa == "anomalia" else 12)
    ax_anom.text(0.98 if tipo_mapa == "anomalia" else 0.97, 0.98 if tipo_mapa == "anomalia" else 0.97, txt_box_anom, transform=ax_anom.transAxes, fontsize=12 if tipo_mapa == "anomalia" else 11, fontweight='bold', color="blue" if mean_anom >= 0 else "red", ha='right', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"), zorder=7)

    for ax in eixos:
        ax.axis('off')
        gdf_recorte.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
        if gdf_rios is not None:
            try: gpd.clip(gdf_rios, gdf_recorte).plot(ax=ax, color='#1f77b4', linewidth=0.7, alpha=0.5, zorder=3)
            except: pass
        if gdf_cid is not None:
            cid_clip = gpd.clip(gdf_cid, gdf_recorte)
            cid_clip.plot(ax=ax, color='red', markersize=35, edgecolor='white', zorder=5)
            textos = [ax.text(r.geometry.x, r.geometry.y, r['nome'], fontsize=10, fontweight='bold', zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]) for _, r in cid_clip.iterrows()]
            try: adjust_text(textos, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            except: pass

    os.makedirs(os.path.dirname(nome_saida), exist_ok=True)
    plt.savefig(nome_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Figura Salva: {os.path.basename(nome_saida)}")

# =============================================================================
# --- 6. MENU INTERATIVO ---
# =============================================================================

def menu_interativo():
    print("="*50)
    print(" 🌍 GERADOR DE PAINÉIS MERGE - MAPAS E ANOMALIAS")
    print("="*50)
    
    # 1. Escala de Tempo
    print("\n[ PASSO 1 ] Escolha a Escala de Tempo:")
    print("  [1] Mensal")
    print("  [2] Trimestral")
    op_escala = input("Digite a opção (padrão=1): ").strip()
    escala_tempo = "trimestral" if op_escala == "2" else "mensal"
    
    # 2. Tipo de Mapa
    print("\n[ PASSO 2 ] Escolha o Tipo de Mapa:")
    print("  [1] Comparativo (Observado Lado a Lado com Anomalia)")
    print("  [2] Apenas Anomalia (Ocupa tela cheia)")
    op_tipo = input("Digite a opção (padrão=1): ").strip()
    tipo_mapa = "anomalia" if op_tipo == "2" else "comparativo"
    
    # 3. Recorte Geográfico
    print("\n[ PASSO 3 ] Escolha o Recorte Geográfico:")
    print("  [1] Estado do Amazonas")
    print("  [2] Bacia Amazônica Completa")
    print("  [3] Apenas Sub-bacias (Negro, Madeira, Solimões)")
    print("  [4] TODOS (Gera os três acima)")
    op_rec = input("Digite a opção (padrão=1): ").strip()
    mapa_recorte = {"1": "amazonas", "2": "bacia", "3": "sub_bacias", "4": "todos"}
    recorte = mapa_recorte.get(op_rec, "amazonas")
    
    # 4. Datas Dinâmicas
    ano_atual = datetime.now().year
    ano_alvo, mes_alvo, trimestre_alvo, ano_fim_trimestre = None, None, None, None
    
    if escala_tempo == "mensal":
        in_ano = input(f"\n[ PASSO 4 ] Digite o ANO (padrão={ano_atual}): ").strip()
        ano_alvo = int(in_ano) if in_ano else ano_atual
        
        mes_atual = datetime.now().month
        in_mes = input(f"Digite o MÊS (1-12) (padrão={mes_atual:02d}): ").strip()
        mes_alvo = int(in_mes) if in_mes else mes_atual
    else:
        in_trim = input("\n[ PASSO 4 ] Digite a sigla do Trimestre (ex: DJF, NDJ) (padrão=DJF): ").strip().upper()
        trimestre_alvo = in_trim if in_trim else "DJF"
        
        in_ano = input(f"Digite o ANO FINAL do trimestre (padrão={ano_atual}): ").strip()
        ano_fim_trimestre = int(in_ano) if in_ano else ano_atual

    print("\n" + "="*50)
    print("Iniciando processamento com as suas configurações...")
    print("="*50 + "\n")
    
    return escala_tempo, tipo_mapa, recorte, ano_alvo, mes_alvo, trimestre_alvo, ano_fim_trimestre

# =============================================================================
# --- 7. ORQUESTRADOR PRINCIPAL ---
# =============================================================================

def main(escala_tempo, tipo_mapa, recorte, ano_alvo, mes_alvo, trimestre_alvo, ano_fim_trimestre):
    
    gdf_bacias = gpd.read_file(SHP_BACIAS).to_crs("epsg:4326")
    if 'wts_cd_p_2' in gdf_bacias.columns: gdf_bacias['wts_cd_p_2'] = gdf_bacias['wts_cd_p_2'].astype(str).str.strip()
    gdf_rios = gpd.read_file(SHP_RIOS).to_crs("epsg:4326")
    gdf_cid = gpd.GeoDataFrame(DF_CIDADES, geometry=gpd.points_from_xy(DF_CIDADES.lon, DF_CIDADES.lat), crs="epsg:4326")

    if escala_tempo == "mensal":
        da_obs_full = carregar_dados_base(encontrar_arquivo(DIR_OBSERVADO, ano_alvo, mes_alvo, "observado"), ano_alvo, mes_alvo, eh_observado_bruto=True)
        da_anom_full = carregar_dados_base(encontrar_arquivo(DIR_ANOMALIA, ano_alvo, mes_alvo, "anomalia"), ano_alvo, mes_alvo)
        titulo_base, lims, p_ano, p_mes, sufixo = f"{NOME_MESES[mes_alvo]} de {ano_alvo}", LIMS_MENSAL, str(ano_alvo), f"{mes_alvo:02d}", f"{mes_alvo:02d}_{ano_alvo}"
    else:
        da_obs_full = obter_dados_trimestre(trimestre_alvo, ano_fim_trimestre, "observado")
        da_anom_full = obter_dados_trimestre(trimestre_alvo, ano_fim_trimestre, "anomalia")
        titulo_base, lims, p_ano, p_mes, sufixo = f"Trimestre {trimestre_alvo} ({ano_fim_trimestre-1}/{ano_fim_trimestre})", LIMS_TRIMES, str(ano_fim_trimestre), trimestre_alvo, f"{trimestre_alvo}_{ano_fim_trimestre}"

    if da_obs_full is None or da_anom_full is None:
        print("❌ ERRO: Faltam arquivos base para o período solicitado. Verifique se os dados MERGE e Anomalias foram calculados na Etapa Anterior.")
        return

    DIR_DEST = DIR_SAIDA_COMP if tipo_mapa == "comparativo" else DIR_SAIDA_ANOM
    pref = "COMPARATIVO_" if tipo_mapa == "comparativo" else "ANOMALIA_"
    os.makedirs(os.path.join(DIR_DEST, p_ano, p_mes), exist_ok=True)

    if recorte in ["amazonas", "todos"]:
        print("--- Gerando Mapa: Amazonas ---")
        gdf_am = gpd.read_file(SHP_AMAZONAS).to_crs("epsg:4326")
        da_obs_am, da_anom_am = processar_recorte(da_obs_full, gdf_am), processar_recorte(da_anom_full, gdf_am)
        if da_obs_am is not None:
            n_saida = os.path.join(DIR_DEST, p_ano, p_mes, f"Amazonas_{pref}{sufixo}.png")
            plotar_mapa(tipo_mapa, da_obs_am, da_anom_am, gdf_am, gdf_rios, gdf_cid, f"Estado do Amazonas - {titulo_base}", n_saida, CAMINHO_DEM, lims)
            exportar_csv_cidades(da_obs_am, da_anom_am, DF_CIDADES, n_saida.replace('.png', '.csv'))

    if recorte in ["bacia", "todos"]:
        print("\n--- Gerando Mapa: Bacia Completa ---")
        da_obs_bac, da_anom_bac = processar_recorte(da_obs_full, gdf_bacias), processar_recorte(da_anom_full, gdf_bacias)
        if da_obs_bac is not None:
            n_saida = os.path.join(DIR_DEST, p_ano, p_mes, f"Geral_{pref}{sufixo}.png")
            plotar_mapa(tipo_mapa, da_obs_bac, da_anom_bac, gdf_bacias, gdf_rios, gdf_cid, f"Bacia Amazônica - {titulo_base}", n_saida, CAMINHO_DEM, lims)
            exportar_csv_cidades(da_obs_bac, da_anom_bac, DF_CIDADES, n_saida.replace('.png', '.csv'))

    if recorte in ["sub_bacias", "todos"]:
        print("\n--- Gerando Mapas: Sub-bacias ---")
        for cod, nome in SUB_BACIAS.items():
            gdf_sub = gpd.read_file(SHP_SUB_SOLIMOES).to_crs("epsg:4326") if nome == "Rio Solimoes" and os.path.exists(SHP_SUB_SOLIMOES) else gdf_bacias[gdf_bacias['wts_cd_p_2'] == cod]
            da_obs_sub, da_anom_sub = processar_recorte(da_obs_full, gdf_sub), processar_recorte(da_anom_full, gdf_sub)
            if da_obs_sub is not None:
                n_saida = os.path.join(DIR_DEST, p_ano, p_mes, f"{nome}_{pref}{sufixo}.png")
                plotar_mapa(tipo_mapa, da_obs_sub, da_anom_sub, gdf_sub, gdf_rios, gdf_cid, f"{nome} - {titulo_base}", n_saida, CAMINHO_DEM, lims)

    print("\n✅ Processo Finalizado com Sucesso!")

if __name__ == "__main__":
    # Chama o menu, captura as escolhas do usuário e joga para o orquestrador
    config = menu_interativo()
    main(*config)