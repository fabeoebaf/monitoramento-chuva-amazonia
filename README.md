# Monitoramento Hidrometeorológico da Bacia Amazônica 🌧️💧

Este repositório contém uma suíte de scripts em Python e Bash para automação do download, processamento espacial e visualização de dados de precipitação na Bacia Amazônica. O sistema utiliza dados do **CHIRPS** e do **MERGE (CPTEC/INPE)** para monitoramento climático, cálculo de anomalias diárias/mensais e geração de boletins operacionais.

## 📁 Estrutura do Repositório

Para manter o repositório leve e organizado, adotamos uma arquitetura que separa o código-fonte dos dados processados. 

* **`/scripts`**: Contém todos os códigos fonte (o único diretório versionado pelo Git).
  * `/chirps_prev`: Download e plotagem espacial do CHIRPS.
  * `/merge`: Scripts para gerenciar dados do MERGE (download GRIB2, conversão via CDO, cálculo unificado de anomalias e geração de mapas).
* **`/data`**: Diretório local **(ignorado pelo Git)** destinado aos dados brutos e shapes.
* **`/outputs`**: Diretório local **(ignorado pelo Git)** onde as figuras `.png` são salvas automaticamente.

---

## ⚙️ Pré-requisitos e Instalação

### 1. Dependências de Sistema
Para o processamento dos dados do MERGE (que chegam em formato GRIB2), é estritamente necessário ter o **CDO (Climate Data Operators)** instalado no seu sistema Linux/WSL:
```bash
sudo apt update
sudo apt install cdo