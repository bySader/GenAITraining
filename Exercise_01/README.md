# GEN AI Upskilling — Exercise 1
## Text Summarization with Groq + JSON Output

---

## Requisitos previos
- Python 3.9 o superior instalado
- VS Code (recomendado)
- Tu API Key de Groq

---

## Pasos para ejecutar el ejercicio

### 1. Crear la carpeta del proyecto
Coloca los tres archivos en una carpeta, por ejemplo:
```
exercise_1/
├── summarizer.py
├── requirements.txt
├── .env.example
└── .env          ← tú lo creas en el paso 3
```

### 2. Instalar dependencias
Abre una terminal en la carpeta del proyecto y ejecuta:
```bash
pip install -r requirements.txt
```

### 3. Configurar tu API Key
Crea un archivo llamado `.env` (copia de `.env.example`) y agrega tu key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Ejecutar la aplicación
```bash
python summarizer.py
```

### 5. Usar la aplicación
1. Selecciona la técnica de prompting: `1`, `2`, o `3`
2. Pega o escribe el texto que deseas resumir
3. Presiona Enter dos veces para enviar
4. Recibe la respuesta en formato JSON

---

## Ejemplo de output esperado
```json
{
  "summary": "La inteligencia artificial está revolucionando industrias clave con diagnósticos médicos y vehículos autónomos, aunque genera dilemas éticos sobre privacidad y empleo.",
  "key_points": [
    "IA transforma medicina y transporte",
    "Mejora eficiencia operativa",
    "Plantea desafíos éticos importantes"
  ],
  "sentiment": "neutral",
  "word_count": 42,
  "technique": "zero-shot"
}
```

---

## Técnicas de prompting implementadas

| # | Técnica | Descripción |
|---|---------|-------------|
| 1 | Zero-shot | Instrucción directa sin ejemplos |
| 2 | Few-shot | Incluye 2 ejemplos del formato esperado |
| 3 | Chain of thought | El modelo razona paso a paso antes de responder |

---

## Modelo usado
`llama3-8b-8192` — rápido, gratuito y disponible en Groq.

Otros modelos disponibles en Groq:
- `llama3-70b-8192` (más preciso)
- `mixtral-8x7b-32768` (mayor contexto)
- `gemma-7b-it`

Para cambiar el modelo, edita esta línea en `summarizer.py`:
```python
model="llama3-8b-8192",
```
