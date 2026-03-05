import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
import rioxarray
from datetime import datetime, timedelta

# --- IMPORTAÇÃO DA BIBLIOTECA ANTI-SOBREPOSIÇÃO ---
try:
    from adjustText import adjust_text
except ImportError:
    print("ERRO: Biblioteca 'adjustText' não encontrada. Instale: pip install adjustText")
    exit()

# =============================================================================
# --- 1. CONFIGURAÇÕES DE DIRETÓRIOS (Caminhos Relativos GitHub) ---
# =============================================================================

# Descobre a pasta atual do script (scripts/chirps_prev) e sobe duas pastas para a raiz
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Diretórios de Entrada e Saída
DIRETORIO_ENTRADA = os.path.join(BASE_DIR, "data", "chirps_prev")
DIRETORIO_SAIDA   = os.path.join(BASE_DIR, "outputs", "chirps_prev")

# Caminhos dos Shapefiles e DEM (Atualizados com suas novas pastas!)
# Nota: Assumi que os nomes dos arquivos lá dentro continuam os mesmos.
SHP_BACIAS       = os.path.join(BASE_DIR, "data", "shapes", "shp_bacias", "BACIA_AMAZONICA_.shp")
SHP_RIOS         = os.path.join(BASE_DIR, "data", "shapes", "shp_rios", "geometrias-corrigidas-Massa-Agua.shp")
SHP_SUB_SOLIMOES = os.path.join(BASE_DIR, "data", "shapes", "shp_sub_solimoes", "sub_solimoes.shp")
CAMINHO_DEM      = os.path.join(BASE_DIR, "data", "shapes", "mde", "bacia.asc")

# =============================================================================

# Configuração das Sub-bacias
SUB_BACIAS = {
    "48": {"nome": "Rio Negro",    "x": 0.03, "y": 0.03, "ha": "left", "va": "bottom"},
    "46": {"nome": "Rio Madeira",  "x": 0.03, "y": 0.97, "ha": "left", "va": "top"},
    "49": {"nome": "Rio Solimoes", "x": 0.97, "y": 0.97, "ha": "right", "va": "top"}
}

# Lista de Cidades (atualizada)
DADOS_CIDADES = [
    {"nome": "Tabatinga",        "lat": -4.2347,   "lon": -69.9447},
    {"nome": "Coari",            "lat": -4.0856,   "lon": -63.0833},
    {"nome": "Manacapuru",       "lat": -3.3106,   "lon": -60.6094},
    {"nome": "Beruri",           "lat": -3.8989,   "lon": -61.3742},
    {"nome": "Lábrea",           "lat": -7.2581,   "lon": -64.7975},
    {"nome": "Rio Branco",       "lat": -9.9758,   "lon": -67.8000},
    {"nome": "Barcelos",         "lat": -0.9658,   "lon": -62.9311},
    {"nome": "Moura",            "lat": -1.4567,   "lon": -61.6347},
    {"nome": "Manaus",           "lat": -3.1383,   "lon": -60.0272},
    {"nome": "Porto Velho",      "lat": -8.7483,   "lon": -63.9169},
    {"nome": "Humaitá",          "lat": -7.5028,   "lon": -63.0183},
    {"nome": "Manicoré",         "lat": -5.8167,   "lon": -61.3019},
    {"nome": "Ji-Paraná",        "lat": -10.8736,  "lon": -61.9356},
    {"nome": "Itacoatiara",      "lat": -3.1539,   "lon": -58.4114},
    {"nome": "Óbidos",           "lat": -1.9192,   "lon": -55.5131},
    {"nome": "Eirunepé",                  "lat": -6.684444,  "lon": -69.8811},
    {"nome": "Ipixuna",                   "lat": -7.050833,  "lon": -71.68417},
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
# --- 2. FUNÇÕES ---

def buscar_ultimo_tif(diretorio_raiz):
    padrao = os.path.join(diretorio_raiz, "**", "*.tif")
    arquivos = glob.glob(padrao, recursive=True)
    if not arquivos: return None
    return max(arquivos, key=os.path.getmtime)

def extrair_data_arquivo(nome_arquivo):
    match = re.search(r"(\d{4})[.-]?(\d{2})[.-]?(\d{2})", nome_arquivo)
    if match: return match.group(1), match.group(2), match.group(3)
    return None, None, None

def calcular_estatisticas(src, geometria):
    try:
        out_image, _ = mask(src, [geometria], crop=True)
        dados = out_image[0].astype('float32')
        dados[dados < -100] = np.nan 
        if np.nanmin(dados) is np.nan: return None
        return np.nanmean(dados), np.nanmin(dados), np.nanmax(dados)
    except Exception:
        return None

def plotar_mapa(src, geometria_recorte, titulo_completo, caminho_salvar, 
                texto_stats=None, cor_texto='black', pos_config=None, 
                gdf_pontos=None, gdf_rios_full=None, dem_path=None,
                gdf_divisoes=None):
    try:
        out_image, out_transform = mask(src, [geometria_recorte], crop=True)
        dados = out_image[0].astype('float32')
        dados[dados < -100] = np.nan 

        if np.nanmin(dados) is np.nan: return

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_extent = plotting_extent(out_image[0], out_transform)
        
        # --- DEM (RELEVO) ---
        hillshade_arr = None
        hillshade_extent = None
        if dem_path and os.path.exists(dem_path):
            try:
                rds = rioxarray.open_rasterio(dem_path)
                if rds.rio.crs is None: rds.rio.write_crs("epsg:4326", inplace=True)
                if rds.rio.crs != src.crs: rds = rds.rio.reproject(src.crs)
                
                rds_clipped = rds.rio.clip([geometria_recorte], src.crs, drop=True)
                dem_data = rds_clipped.isel(band=0).values.astype('float32')
                dem_data = np.where(dem_data < -500, np.nan, dem_data)
                
                ls = LightSource(azdeg=315, altdeg=45)
                hillshade_arr = ls.hillshade(dem_data, vert_exag=2000)
                b = rds_clipped.rio.bounds()
                hillshade_extent = [b[0], b[2], b[1], b[3]]
            except: pass

        if hillshade_arr is not None:
            ax.imshow(hillshade_arr, cmap='gray', extent=hillshade_extent, origin='upper', alpha=1.0, zorder=1)

        # --- CONFIGURAÇÃO DA BARRA DE CORES (10 em 10 mm) ---
        vmin_user, vmax_user = -60, 60
        passo_user = 10
        ticks_bar = np.arange(vmin_user, vmax_user + 1, passo_user)
        # ----------------------------------------------------

        img_plot = ax.imshow(dados, cmap=plt.cm.BrBG, 
                             vmin=vmin_user, vmax=vmax_user, 
                             extent=plot_extent, zorder=2, alpha=0.7)
        
        # Divisões Internas
        if gdf_divisoes is not None:
            gs_recorte = gpd.GeoSeries([geometria_recorte], crs=src.crs)
            divisoes_clip = gpd.clip(gdf_divisoes, gs_recorte)
            divisoes_clip.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6, zorder=4, linestyle='--')

        # Contorno
        gs_geom = gpd.GeoSeries([geometria_recorte], crs=src.crs)
        gs_geom.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
        
        # Rios
        if gdf_rios_full is not None:
            if gdf_rios_full.crs != src.crs: gdf_rios_full = gdf_rios_full.to_crs(src.crs)
            rios_clip = gpd.clip(gdf_rios_full, gs_geom)
            if not rios_clip.empty:
                rios_clip.plot(ax=ax, color='blue', linewidth=1.3, zorder=3)
        
        # Cidades
        if gdf_pontos is not None:
            if gdf_pontos.crs != src.crs: pontos_proj = gdf_pontos.to_crs(src.crs)
            else: pontos_proj = gdf_pontos
            pontos_clip = gpd.clip(pontos_proj, gs_geom)
            if not pontos_clip.empty:
                pontos_clip.plot(ax=ax, color='red', markersize=35, edgecolor='white', linewidth=0.8, zorder=5)
                textos = []
                for _, row in pontos_clip.iterrows():
                    t = ax.text(row.geometry.x, row.geometry.y, row['nome'], 
                                fontsize=9, color='black', fontweight='bold', zorder=6,
                                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
                    textos.append(t)
                if textos: adjust_text(textos, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, zorder=4))

        # --- BARRA DE CORES HORIZONTAL (EM BAIXO) ---
        cbar = plt.colorbar(img_plot, ax=ax, 
                            orientation='horizontal',  # Define horizontal
                            shrink=0.75,               # Largura da barra em relação ao eixo
                            pad=0.05,                  # Espaço entre o mapa e a barra (0.05 é bom para baixo)
                            ticks=ticks_bar,
                            aspect=40)                 # Controla a "espessura" (quanto maior, mais fina)
        
        cbar.set_label('Anomalia (mm)', fontsize=12)
        cbar.ax.tick_params(labelsize=9)
        # --------------------------------
        
        ax.set_title(titulo_completo, fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')

        if texto_stats and pos_config:
            ax.text(pos_config['x'], pos_config['y'], texto_stats,
                    transform=ax.transAxes, fontsize=11, fontweight='bold', color=cor_texto,
                    verticalalignment=pos_config['va'], horizontalalignment=pos_config['ha'],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", linewidth=0.5, alpha=0.85), zorder=7)
        
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> Figura gerada: {os.path.basename(caminho_salvar)}")
        
    except Exception as e:
        print(f"   [ERRO NO PLOT]: {e}")
        import traceback
        traceback.print_exc()

# --- 3. EXECUÇÃO ---

arquivo_tif = buscar_ultimo_tif(DIRETORIO_ENTRADA)
if not arquivo_tif: 
    print("Nenhum arquivo encontrado em:", DIRETORIO_ENTRADA)
    exit()

nome_arq = os.path.basename(arquivo_tif)
ano, mes, dia = extrair_data_arquivo(nome_arq)
if not ano: exit()

dt_ini = datetime(int(ano), int(mes), int(dia))
dt_fim = dt_ini + timedelta(days=5)
texto_periodo = f"{dt_ini.strftime('%d/%m')} à {dt_fim.strftime('%d/%m')}"

print(f"Processando: {texto_periodo}")

pasta_destino = os.path.join(DIRETORIO_SAIDA, ano, mes, dia)
os.makedirs(pasta_destino, exist_ok=True)
arquivo_txt = open(os.path.join(pasta_destino, f"Stats_{ano}-{mes}-{dia}.txt"), "w")

print("Carregando Shapes...")
gdf_bacias = gpd.read_file(SHP_BACIAS)
gdf_bacias['wts_cd_p_2'] = gdf_bacias['wts_cd_p_2'].astype(str).str.strip()
gdf_rios = gpd.read_file(SHP_RIOS)

df_cidades = pd.DataFrame(DADOS_CIDADES)
gdf_cidades = gpd.GeoDataFrame(df_cidades, geometry=gpd.points_from_xy(df_cidades.lon, df_cidades.lat), crs="EPSG:4326")

with rasterio.open(arquivo_tif) as src:
    if gdf_bacias.crs != src.crs: gdf_bacias = gdf_bacias.to_crs(src.crs)
    if gdf_rios.crs != src.crs: gdf_rios = gdf_rios.to_crs(src.crs)

    # 1. GERAL
    geometria_geral_uniao = gdf_bacias.union_all() 
    stats = calcular_estatisticas(src, geometria_geral_uniao)
    if stats:
        media, _, _ = stats
        seta, cor = ("↑", "blue") if media >= 0 else ("↓", "red")
        arquivo_txt.write(f"Geral: Média {media:.1f}\n")
        
        config_geral = {"x": 0.97, "y": 0.97, "ha": "right", "va": "top"}
        titulo = f"Bacia Amazônica\n{texto_periodo}"
        
        plotar_mapa(src, geometria_geral_uniao, titulo, 
                    os.path.join(pasta_destino, f"Geral_{ano}-{mes}-{dia}.png"),
                    texto_stats=f"Média: {media:.1f} mm {seta}", 
                    cor_texto=cor, pos_config=config_geral,
                    gdf_pontos=gdf_cidades, gdf_rios_full=gdf_rios,
                    dem_path=CAMINHO_DEM,
                    gdf_divisoes=gdf_bacias)

    # 2. SUB-BACIAS
    for codigo, info in SUB_BACIAS.items():
        nome_rio = info["nome"]
        
        if nome_rio == "Rio Solimoes" and os.path.exists(SHP_SUB_SOLIMOES):
            gdf_divisoes_rio = gpd.read_file(SHP_SUB_SOLIMOES)
            if gdf_divisoes_rio.crs != src.crs: gdf_divisoes_rio = gdf_divisoes_rio.to_crs(src.crs)
            geometria_bacia = gdf_divisoes_rio.union_all()
            gdf_para_plotar = gdf_divisoes_rio
        else:
            sub_gdf = gdf_bacias[gdf_bacias['wts_cd_p_2'] == codigo]
            if sub_gdf.empty: continue
            geometria_bacia = sub_gdf.geometry.iloc[0]
            gdf_para_plotar = None

        stats = calcular_estatisticas(src, geometria_bacia)
        if stats:
            media, _, _ = stats
            seta, cor = ("↑", "blue") if media >= 0 else ("↓", "red")
            arquivo_txt.write(f"{nome_rio}: Média {media:.1f}\n")
            
            titulo = f"{nome_rio}\n{texto_periodo}"
            
            plotar_mapa(src, geometria_bacia, titulo, 
                        os.path.join(pasta_destino, f"{nome_rio.replace(' ', '_')}_{ano}-{mes}-{dia}.png"),
                        texto_stats=f"Média: {media:.1f} mm {seta}", 
                        cor_texto=cor, pos_config=info,
                        gdf_pontos=gdf_cidades, gdf_rios_full=gdf_rios,
                        dem_path=CAMINHO_DEM,
                        gdf_divisoes=gdf_para_plotar)

arquivo_txt.close()
print("Processo Finalizado.")