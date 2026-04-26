# 📡 ChurnRadar — Previsão de Churn com XGBoost + Deploy

Projeto completo de Machine Learning com deploy em produção: do treinamento do modelo até um site acessível publicamente, passando por API REST na nuvem.

**🌐 Site:** [carloseadi.github.io/Churn-Deploy](https://carloseadi.github.io/Churn-Deploy)  
**🔌 API:** [churn-api-nltm.onrender.com](https://churn-api-nltm.onrender.com)  
**📖 Docs:** [churn-api-nltm.onrender.com/docs](https://churn-api-nltm.onrender.com/docs)

---

## 🏗️ Arquitetura

```
Usuário → Site (GitHub Pages) → API REST (Render) → Modelo XGBoost (.pkl)
```

---

## 📁 Estrutura do Projeto

```
churn_deploy/
├── train.py            # Treinamento do modelo XGBoost
├── main.py             # API FastAPI
├── churn_model.pkl     # Modelo serializado (gerado pelo train.py)
├── index.html          # Frontend (hospedado no GitHub Pages)
├── requirements.txt    # Dependências Python
└── Procfile            # Configuração de inicialização para o Render
```

---

## 🤖 Modelo

- **Algoritmo:** XGBoost Classifier
- **Problema:** Classificação binária (churn / não churn)
- **Dataset:** Sintético com 5.000 clientes
- **ROC-AUC:** 0.71

### Features utilizadas

| Feature | Descrição |
|---|---|
| `tenure_months` | Tempo como cliente (meses) |
| `monthly_charges` | Valor cobrado mensalmente |
| `num_products` | Quantidade de produtos contratados |
| `support_calls` | Número de chamadas ao suporte |
| `has_contract` | Possui contrato ativo (0 ou 1) |
| `payment_delay_days` | Dias de atraso no pagamento |
| `satisfaction_score` | Nota de satisfação (1 a 5) |

### Feature Importances

```
has_contract          0.2858  ███████████
satisfaction_score    0.1551  ██████
tenure_months         0.1508  ██████
monthly_charges       0.1333  █████
support_calls         0.1099  ████
payment_delay_days    0.0889  ███
num_products          0.0763  ███
```

---

## 🚀 Como rodar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/carloseadi/Churn-Deploy.git
cd Churn-Deploy
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python train.py
```

Isso gera o arquivo `churn_model.pkl`.

### 4. Subir a API

```bash
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`.  
Documentação interativa: `http://localhost:8000/docs`

---

## 🔌 Endpoints da API

### `GET /`
Health check da API.

```json
{
  "status": "ok",
  "modelo": "XGBoost Churn",
  "features": ["tenure_months", "monthly_charges", ...]
}
```

### `POST /predict`
Previsão para um único cliente.

**Request:**
```json
{
  "tenure_months": 8,
  "monthly_charges": 95.0,
  "num_products": 1,
  "support_calls": 5,
  "has_contract": 0,
  "payment_delay_days": 15,
  "satisfaction_score": 2
}
```

**Response:**
```json
{
  "churn": true,
  "probabilidade_churn": 0.8681,
  "risco": "Alto"
}
```

### `POST /predict/batch`
Previsão para múltiplos clientes (máximo 100).

---

## ☁️ Deploy

### API — Render
A API está hospedada no [Render](https://render.com) (plano gratuito).  
O Render detecta automaticamente qualquer push na branch `main` e faz redeploy.

> ⚠️ No plano gratuito o servidor dorme após 15 minutos de inatividade. A primeira requisição pode demorar ~50 segundos para acordar.

### Frontend — GitHub Pages
O site `index.html` é hospedado via GitHub Pages e consome a API diretamente do browser via `fetch`.

---

## 🛠️ Tecnologias

| Camada | Tecnologia |
|---|---|
| Modelo | XGBoost, scikit-learn |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Hospedagem API | Render |
| Hospedagem Site | GitHub Pages |
| Serialização | Pickle |

---

## 📈 Próximos passos

- [ ] Substituir dataset sintético por dados reais
- [ ] Adicionar autenticação na API (API Key)
- [ ] Logar predições em banco de dados
- [ ] Pipeline de retreino automático
- [ ] Dashboard de monitoramento do modelo
