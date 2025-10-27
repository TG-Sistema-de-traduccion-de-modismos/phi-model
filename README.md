# **Phi Model Service**

**Servicio instruccional para neutralización de modismos** usando `microsoft/Phi-3-mini-4k-instruct`. Este servicio actúa como un traductor de modismos a lenguaje natural, recibiendo frases con modismos y sus significados para generar una versión neutral y gramaticalmente correcta. Imagen optimizada para **RTX 5070**.

## **Resumen**
Este servicio:
1. Recibe una frase que contiene modismos
2. Recibe un diccionario de modismos y sus significados neutrales
3. Genera una nueva frase reemplazando los modismos por sus equivalentes en lenguaje neutral
4. Asegura que la frase resultante sea gramaticalmente correcta

## **Características principales**
- **Modelo base:** `microsoft/Phi-3-mini-4k-instruct`
- **Función:** Neutralización de modismos a lenguaje natural
- **Entrada:** Frase original + diccionario de significados
- **Salida:** Frase neutralizada gramaticalmente correcta

## **Estructura del proyecto**
```
phi-model/
├── Dockerfile
├── requirements.txt
├── .env                    # MODEL_NAME, HOST, PORT, DEVICE, TORCH_DTYPE
└── app/
    ├── main.py            # Punto de entrada FastAPI
    ├── infrastructure/
    │   └── routes.py      # Endpoints HTTP
    ├── application/
    │   └── model_wrapper.py  # Wrapper para Phi-3
    ├── core/
    │   ├── config.py      # Configuración
    │   └── logging_config.py
    └── domain/
        └── models.py      # Modelos Pydantic
```

## **Endpoints**

### **GET /health**
Endpoint de diagnóstico que verifica:
- Si el modelo está cargado correctamente en memoria
- El dispositivo en uso (GPU/CPU)
- El nombre exacto del modelo cargado
- Estado general del servicio

Retorna:
```json
{
    "status": "ok",
    "model_loaded": true,
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "device": "cuda"
}
```

### **POST /neutralizer**
Endpoint principal que:
1. Recibe una frase con modismos y sus significados correspondientes
2. Procesa la frase utilizando el modelo Phi-3
3. Genera una nueva versión donde:
   - Reemplaza cada modismo por su significado neutral
   - Ajusta la gramática para mantener coherencia
   - Preserva el significado original de la frase
4. Retorna la frase neutralizada con metadatos del proceso

**Proceso**

Recibe:
```json
{
    "frase": "Parchamos ayer con los amigos",
    "significado": {
        "parchar": "salir"
    }
}
```

Retorna:
```json
{
    "neutralizada": "salimos ayer con los amigos",
    "modelo": "microsoft/Phi-3-mini-4k-instruct",
    "tiempo_proceso": 0.234
}
```


## **Docker — build & run usando Dockerfile y GPU**
Requisitos en host: Docker 19.03+ y **NVIDIA Container Toolkit**. Comandos desde PowerShell o CMD en Windows:

1) Construir la imagen:
```sh
docker build -t phi-model:latest ./phi-model
```
> **Nota:** la imagen resultante pesa aproximadamente **28.33 GB**.

> **Nota:** esta imagen de Docker fue especialmente hecha para su uso con una RTX 5070.

2) Ejecutar (opción moderna --gpus, recomendada):
```sh
docker run --rm --name phi-model `
    --gpus all `
    -e NVIDIA_VISIBLE_DEVICES=all `
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
    --env-file .env `
    -p 8006:8006 `
    phi-model:latest
```


## **Configuración importante**
- **Revisa `requirements.txt`** y usa las versiones indicadas (torch, transformers, tokenizers, etc.). Las incompatibilidades entre versiones suelen ser la causa principal de errores en carga o inferencia.  
- Si usas archivos de configuración o `.env`, **actualiza las IPs/hosts/puertos** en `app/core/config.py` y en `.env` si tu despliegue no usa `localhost` (p. ej. `SERVICE_HOST`, `MODEL_HOST`).  
---

## **Limitaciones y recomendaciones**
- El repositorio solo **carga** y **sirve** el modelo fine‑tuned; **no incluye** el pipeline de entrenamiento.  
- Ejecutar en GPU con suficiente VRAM para evitar OOM al cargar modelos grandes.  
- Montar el checkpoint externamente si quieres reducir el tamaño de la imagen o facilitar actualizaciones sin rebuild.  
- Revisar logs (logger) para diagnosticar errores de carga o inferencia.ld
- Actualizar IPs/puertos en configuración si no se usa localhost
