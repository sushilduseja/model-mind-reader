# Model Mind Reader — Frontend

Next.js frontend for the Model Mind Reader ML interpretability tool.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Home — navigation to all features |
| `/data-ingestion` | Upload CSV, select target column & model type, train model |
| `/model-training` | Model training interface |
| `/explainability` | Generate LIME/SHAP explanations |
| `/fairness-metrics` | Evaluate fairness metrics |

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend must be running at `http://127.0.0.1:8000`.

## Build

```bash
npm run build
npm start
```
