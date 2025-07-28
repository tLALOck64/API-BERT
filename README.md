
# API-BERT

Este proyecto implementa modelos de procesamiento de lenguaje natural (NLP) utilizando BERT y BART para la identificación de intenciones y el análisis de sentimientos en comentarios de usuarios de la plataforma Nativox.

## Descripción

La API permite analizar comentarios de usuarios, clasificando su intención (por ejemplo, apreciación, queja, solicitud, reporte de error o confusión) y determinando el sentimiento (positivo, neutro o negativo) mediante modelos preentrenados de Hugging Face. Además, integra funcionalidades para la gestión de archivos en AWS S3 y la interacción con servicios externos.

## Características principales
- Análisis de sentimiento usando BERT Multilingual.
- Clasificación de intención usando BART (zero-shot classification).
- Limpieza y normalización avanzada de texto.
- Procesamiento masivo de datasets de comentarios.
- Exportación de resultados y resúmenes en formato CSV.
- Integración con AWS S3 para almacenamiento de archivos.
- Consumo de datos desde APIs externas.

## Estructura del proyecto

```
app/
    main.py
    api/
        data_router.py
    repositories/
        external_api.py
    services/
        sentiment_analysis_service.py
        data_service.py
        ...
datasets/
tools/
    diccionario_abrevaciones.csv
.env.example
.gitignore
```

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/tLALOck64/API-BERT.git
   cd API-BERT
   ```
2. Crea y activa un entorno virtual:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Configura las variables de entorno en el archivo `.env`.

## Ejecución

### API principal (FastAPI)

Desde la raíz del proyecto, ejecuta:
```
uvicorn app.main:app --reload
```

## Uso

- Accede a la documentación interactiva en `http://localhost:8000/docs` para probar los endpoints disponibles.
- Los endpoints principales se encuentran bajo el prefijo `/data`.

## Notas
- Los archivos de configuración y credenciales (`.env`, `.venv`, `datasets/`, `__pycache__/`) están excluidos del control de versiones mediante `.gitignore`.
- Asegúrate de contar con las credenciales necesarias para AWS y Hugging Face.

## Licencia

Este proyecto es de uso académico y no está destinado para producción sin las debidas adaptaciones de seguridad y escalabilidad.
