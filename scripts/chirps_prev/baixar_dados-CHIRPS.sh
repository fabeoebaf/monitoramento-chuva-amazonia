#!/bin/bash

# =============================================================================
# --- CONFIGURAÇÕES DE DIRETÓRIO (Caminhos Relativos para o GitHub) ---
# =============================================================================

# Descobre a pasta exata onde este script está salvo (scripts/chirps_prev)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sobe duas pastas para chegar na raiz do repositório
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Define onde os dados serão salvos (NÃO vão para o GitHub, ficam no /data/)
DIRETORIO_RAIZ="${BASE_DIR}/data/chirps_prev"

# =============================================================================
# --- CONFIGURAÇÕES DE DOWNLOAD ---
# =============================================================================

# Data atual para montar a URL base
ANO_ATUAL=$(date +%Y)

# URL da Fonte
URL_FONTE="https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/05_day/global/anom/${ANO_ATUAL}/"

# Cores para visualização
VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
SEM_COR='\033[0m'

echo -e "${AZUL}=== SINCRONIZANDO ÚLTIMO ARQUIVO (${ANO_ATUAL}) ===${SEM_COR}"
echo -e "Fonte: $URL_FONTE"
echo -e "Destino Local: $DIRETORIO_RAIZ"

# 1. IDENTIFICAR O ARQUIVO MAIS RECENTE
echo "Buscando lista de arquivos..."
ULTIMO_ARQUIVO=$(wget -q -O - "$URL_FONTE" | grep -o 'href="[^"]*\.tif"' | cut -d'"' -f2 | sort | tail -n 1)

if [ -z "$ULTIMO_ARQUIVO" ]; then
    echo -e "${VERMELHO}Erro: Nenhum arquivo .tif encontrado em ${URL_FONTE}${SEM_COR}"
    exit 1
fi

echo -e "Arquivo mais recente detectado: ${AMARELO}${ULTIMO_ARQUIVO}${SEM_COR}"

# 2. EXTRAIR A DATA DO NOME DO ARQUIVO
# O Regex procura: 4 digitos (ano) + separador opcional + 2 digitos (mês)
if [[ $ULTIMO_ARQUIVO =~ ([0-9]{4})[.-]?([0-9]{2})[.-]?([0-9]{2}) ]]; then
    ANO_ARQ="${BASH_REMATCH[1]}"
    MES_ARQ="${BASH_REMATCH[2]}"
    
    # 3. PREPARAR DESTINO LOCAL
    # Cria a estrutura de pastas localmente separada por ano e mês
    PASTA_DESTINO="${DIRETORIO_RAIZ}/${ANO_ARQ}/${MES_ARQ}"
    CAMINHO_FINAL="${PASTA_DESTINO}/${ULTIMO_ARQUIVO}"
    
    mkdir -p "$PASTA_DESTINO"

    # 4. BAIXAR (Se não existir)
    if [ -f "$CAMINHO_FINAL" ]; then
        echo -e "${VERDE}O arquivo já existe na pasta ${MES_ARQ}/${ANO_ARQ}. Nada a fazer.${SEM_COR}"
        exit 0
    else
        echo -e "Baixando para: ${PASTA_DESTINO}"
        # Monta a URL completa juntando a base do ano + nome do arquivo
        wget -q --show-progress -O "$CAMINHO_FINAL" "${URL_FONTE}${ULTIMO_ARQUIVO}"
        
        if [ $? -eq 0 ]; then
            echo -e "${VERDE}Download concluído com sucesso!${SEM_COR}"
        else
            echo -e "${VERMELHO}Falha no download.${SEM_COR}"
            # Remove arquivo vazio caso tenha criado
            rm -f "$CAMINHO_FINAL"
        fi
    fi

else
    echo -e "${VERMELHO}Erro: Não consegui identificar o mês no nome do arquivo: $ULTIMO_ARQUIVO${SEM_COR}"
    echo "O arquivo será baixado na raiz do ano para segurança."
    mkdir -p "${DIRETORIO_RAIZ}/${ANO_ATUAL}"
    wget -q --show-progress -P "${DIRETORIO_RAIZ}/${ANO_ATUAL}" "${URL_FONTE}${ULTIMO_ARQUIVO}"
fi