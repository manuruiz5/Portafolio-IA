# 🎯 Agente de Soporte Técnico con RAG y Gestión de Estado

## Contexto

Este desafío integrador consolida todos los conceptos aprendidos en la práctica de LangGraph para construir un **agente de soporte técnico funcional** para StreamMax, una plataforma ficticia de streaming. El sistema debe manejar dos tipos de consultas fundamentales: (1) **preguntas sobre el producto** que requieren búsqueda en documentación mediante RAG, y (2) **consultas de estado de cuenta** que acceden a información personalizada del usuario mediante tools dedicadas.

El caso de uso simula un escenario real donde los usuarios alternan entre preguntas generales ("¿cuáles son los planes?") y consultas específicas de su cuenta ("¿cuál es mi estado de suscripción?"). El agente debe **decidir autónomamente** qué herramienta usar en cada turno, mantener contexto conversacional entre turnos, y combinar información de múltiples fuentes para respuestas complejas (ej: "si cancelo ahora según mi plan actual, ¿cuándo expira el servicio?").

---

## ⚙️ Objetivos

* **Construir RAG especializado** con 8 documentos sobre planes, dispositivos, troubleshooting y políticas de StreamMax.
* **Implementar tool de estado** que consulte base de datos ficticia de suscripciones por user_id.
* **Diseñar grafo robusto** assistant ↔ tools con routing condicional para decisiones dinámicas.
* **Probar 3 escenarios**: (1) Solo RAG, (2) Solo tool de estado, (3) Multi-turno combinando ambas.
* **Validar cumplimiento** de requisitos: AgentState completo, 2 tools funcionales, logs de herramientas usadas.

---

## 📋 Actividades

| Actividad | Descripción | Resultado Obtenido |
| :--- | :--- | :--- |
| **1. Creación de base de conocimiento** | Corpus de 8 documentos sobre StreamMax: planes ($9.99-$19.99), dispositivos compatibles (Smart TV, móviles, consolas), troubleshooting (buffering, descargas offline), políticas (cancelación, perfiles). | **8 chunks indexados** en FAISS con embeddings OpenAI. Retriever configurado para k=3 documentos más relevantes. |
| **2. Tool RAG (search_docs)** | Decorador `@tool` convierte retriever en función llamable. Busca en documentación por similitud semántica y retorna contexto concatenado. | Tool `search_docs` funcional. Maneja casos sin resultados con mensaje: "*No encontré información relevante...*". |
| **3. Tool de estado (get_subscription_status)** | Base de datos dummy con 3 usuarios (user123: Premium activa, user456: Básico activa, user789: Estándar cancelada). Retorna plan, estado, próxima facturación y dispositivos activos. | Tool `get_subscription_status` funcional. Formato legible con emojis y estructura clara. |
| **4. Configuración de grafo** | AgentState con `messages` + `summary`. Grafo con nodos assistant/tools, routing condicional basado en `tool_calls`, ciclo assistant→tools→assistant. | Grafo compilado correctamente. LLM con 2 tools bindeadas: `search_docs` + `get_subscription_status`. |
| **5. Conversación 1: Solo RAG** | Query: "*¿Cuáles son los planes de StreamMax y sus precios?*". Agente debe usar solo `search_docs`. | ✅ **Tool usada: search_docs**. Respuesta completa con 3 planes (Básico $9.99, Estándar $14.99, Premium $19.99) + info de dispositivos y contenido original. |
| **6. Conversación 2: Solo estado** | Query: "*Necesito saber el estado de mi suscripción. Mi usuario es user123.*". Agente debe usar solo `get_subscription_status`. | ✅ **Tool usada: get_subscription_status**. Respuesta: Plan Premium activo, próxima facturación 03/12/2025, 3 dispositivos activos. |
| **7. Conversación 3: Multi-turno** | **Turno 1**: Dispositivos compatibles (RAG). **Turno 2**: Estado user456 (Tool). **Turno 3**: Política de cancelación basada en contexto previo (RAG + memoria). | ✅ **Tools usadas: search_docs + get_subscription_status**. El agente mantiene contexto: en T3 responde "*hasta el 26 de noviembre*" (fecha de la facturación de user456 obtenida en T2). |
| **8. Validación de requisitos** | Verificar cumplimiento de todos los requisitos mínimos del desafío. | ✅ **100% completo**: AgentState con messages+summary, RAG con 8 docs, 2 tools funcionales, grafo assistant↔tools, 3 conversaciones con logs. |

---

## Desarrollo

### 📚 Actividad 1-2: Base de Conocimiento y Tool RAG

**Corpus diseñado:**

- **Planes y precios**: 3 tiers con características diferenciadas (SD/HD/4K, 1-4 pantallas)
- **Compatibilidad**: 5 categorías de dispositivos (Smart TV, navegadores, móviles, streaming boxes, consolas)
- **Funcionalidades**: Descargas offline (límite 100 títulos, 48h expiración), perfiles (5 máximo, Kids con control parental)
- **Troubleshooting**: Soluciones para buffering (velocidad mínima 5-25 Mbps, reinicio de router)
- **Políticas**: Cancelación (efectiva fin de período), recuperación de contraseña

**Arquitectura técnica:**

```
8 documentos → RecursiveCharacterTextSplitter (chunk_size=500) 
           → OpenAIEmbeddings 
           → FAISS (8 chunks, k=3)
```

**Tool `search_docs`:**

- Entrada: `question` (str) con query del usuario
- Proceso: Similarity search por embeddings coseno
- Salida: Concatenación de top-3 chunks o mensaje de fallback

**Resultado:** Base de conocimiento funcional que cubre casos típicos de soporte L1 (preguntas frecuentes, configuración básica).

---

### 🛠️ Actividad 3-4: Tool de Estado y Grafo

**Base de datos de suscripciones:**

| User ID | Plan | Estado | Próxima Facturación | Dispositivos |
|---------|------|--------|---------------------|--------------|
| user123 | Premium | Activa | 2025-12-03 | 3 |
| user456 | Básico | Activa | 2025-11-26 | 1 |
| user789 | Estándar | Cancelada | N/A | 0 |

**Tool `get_subscription_status`:**

- Validación de `user_id` contra diccionario
- Formato de respuesta con emojis (📊) para mejor UX
- Manejo de errores: mensaje claro si user_id no existe

**Grafo configurado:**

```
START → assistant → ¿tool_calls?
                    ├─ Sí → tools → assistant (ciclo)
                    └─ No → END
```

**Decisión de diseño:** No se implementó el nodo `memory` en este desafío porque las 3 conversaciones son independientes. En producción, se activaría después de 3+ turnos.

---

### 💬 Actividad 5: Conversación 1 - Solo RAG

**Query del usuario:**

> "Hola, ¿cuáles son los planes de StreamMax y sus precios?"

**Análisis del comportamiento del agente:**

1. **Decisión del LLM:** Detecta que la pregunta es sobre información del producto → Llama `search_docs`
2. **Ejecución de tool:** FAISS retorna chunk sobre planes ($9.99-$19.99 con detalles)
3. **Síntesis:** El agente combina información de múltiples chunks:

   - Planes y precios (chunk 1)
   - Dispositivos compatibles (chunk 3)
   - Contenido original (chunk 8)

**Respuesta generada:**

> "StreamMax ofrece los siguientes planes:

> 1. **Básico**: $9.99/mes, calidad SD, 1 pantalla.
> 2. **Estándar**: $14.99/mes, calidad HD, 2 pantallas.
> 3. **Premium**: $19.99/mes, calidad 4K, 4 pantallas.
> 
> Además, StreamMax es compatible con... [lista de dispositivos]... También producen contenido original exclusivo..."

**Observación clave:** El agente **agregó información no solicitada** (dispositivos, contenido) porque el retriever devolvió esos chunks. Esto es típico de RAG: a veces da contexto extra útil, otras veces puede ser ruido.

**Tool usada:** `search_docs` ✅

---

### 💬 Actividad 6: Conversación 2 - Solo Tool de Estado

**Query del usuario:**

> "Necesito saber el estado de mi suscripción. Mi usuario es user123."

**Análisis del comportamiento del agente:**

1. **Decisión del LLM:** Detecta query de estado personalizado + `user_id` explícito → Llama `get_subscription_status`
2. **Ejecución de tool:** Consulta diccionario y retorna datos de user123
3. **Formateo:** El agente presenta la información de forma estructurada

**Respuesta generada:**

> "El estado de tu suscripción es el siguiente:

> - **Plan:** Premium
> - **Estado:** Activa
> - **Próxima facturación:** 2025-12-03
> - **Dispositivos activos:** 3
> 
> Si necesitas más información, no dudes en preguntar."

**Observación clave:** El agente **no usó RAG** a pesar de tener esa tool disponible. Esto demuestra que el LLM está tomando decisiones correctas basadas en las descripciones de las tools.

**Tool usada:** `get_subscription_status` ✅

---

### 💬 Actividad 7: Conversación 3 - Multi-turno con Contexto

Esta es la conversación más compleja porque requiere:

- Mantener estado entre 3 turnos
- Usar ambas tools en momentos distintos
- Combinar información de turnos anteriores para responder

#### **Turno 1: Dispositivos compatibles**

**Query:** "*¿En qué dispositivos puedo ver StreamMax?*"

**Comportamiento:**

- LLM llama `search_docs`
- Retriever devuelve chunk sobre compatibilidad
- Respuesta lista 5 categorías de dispositivos + info de descargas offline

**Tool usada:** `search_docs`

#### **Turno 2: Estado de suscripción**

**Query:** "*Perfecto. Ahora consultá mi suscripción, soy user456.*"

**Comportamiento:**

- LLM llama `get_subscription_status` con `user_id=user456`
- Retorna: Plan Básico, Activa, próxima facturación **26 de noviembre de 2025**, 1 dispositivo

**Tool usada:** `get_subscription_status`

**Clave:** El agente **guarda esta fecha en el contexto** (en `state["messages"]`).

#### **Turno 3: Política de cancelación contextual**

**Query:** "*Si cancelo ahora, ¿cuándo deja de funcionar?*"

**Comportamiento esperado:**

1. LLM debe usar `search_docs` para obtener política de cancelación
2. Combinar esa info con la fecha de facturación de user456 (del turno 2)

**Respuesta generada:**

> "Si cancelas tu suscripción ahora, seguirás teniendo acceso a StreamMax hasta la fecha de tu próxima facturación, que es el **26 de noviembre de 2025**. Después de esa fecha, tu suscripción se cancelará..."

**Análisis crítico:**

✅ **Éxito 1:** El agente **recordó la fecha del turno 2** sin necesidad de llamar nuevamente a `get_subscription_status`

✅ **Éxito 2:** Combinó política general (de RAG) con dato específico del usuario (de tool de estado)


---

### 📊 Actividad 8: Validación de Requisitos

**Checklist de cumplimiento:**

| Requisito | Estado | Evidencia |
|-----------|--------|-----------|
| AgentState con `messages` | ✅ | Configurado en CELDA 4 con `Annotated[list, operator.add]` |
| AgentState con `summary` | ✅ | Definido como `Optional[str]`, inicializado en `None` |
| RAG con 5-10 textos del dominio | ✅ | 8 documentos sobre StreamMax indexados en FAISS |
| Tool `rag_search` (o similar) | ✅ | Implementada como `search_docs` con decorador `@tool` |
| Tool de estado | ✅ | Implementada como `get_subscription_status` con BD ficticia |
| Grafo assistant ↔ tools | ✅ | Compilado con routing condicional basado en `tool_calls` |
| 3 conversaciones probadas | ✅ | Conv1: Solo RAG, Conv2: Solo estado, Conv3: Multi-turno |
| Logs de tools usadas | ✅ | Mostrados en cada conversación con extracción de `tool_calls` |

**Resultado:** **100% de los requisitos cumplidos** ✅

---

## 💭 Reflexión

Este desafío integrador demuestra tres capacidades críticas de los agentes conversacionales modernos:

### 1. **Routing inteligente sin reglas explícitas**

A diferencia de chatbots tradicionales con árboles de decisión hardcodeados, este agente **infiere la herramienta correcta** de las descripciones en lenguaje natural:

- "*Busca información en la documentación de StreamMax...*" → `search_docs`
- "*Consulta el estado de suscripción de un usuario...*" → `get_subscription_status`

El LLM actúa como un **router semántico** que mapea intenciones a acciones sin necesidad de patrones regex o keywords.

### 2. **Memoria implícita mediante contexto conversacional**

El turno 3 de la conversación 3 es revelador: el agente **no volvió a llamar a `get_subscription_status`** para obtener la fecha de facturación. En cambio, la extrajo del historial de mensajes (`state["messages"]`).

Esto es posible porque:

- El `AgentState` acumula mensajes con `operator.add`
- El LLM recibe **todo el historial** en cada invocación
- GPT-4o-mini tiene suficiente capacidad de contexto (128K tokens) para recordar turnos previos

**Implicación para producción:** En conversaciones largas (20+ turnos), este enfoque falla. Ahí es donde el nodo `memory` con resúmenes se vuelve esencial.

### 3. **Composición de información de múltiples fuentes**

La pregunta "*Si cancelo ahora, ¿cuándo deja de funcionar?*" requiere:

1. **Política general** (de RAG): "La cancelación es efectiva al final del período de facturación"
2. **Dato específico** (de tool): "Tu próxima facturación es el 26 de noviembre"
3. **Síntesis**: "...hasta el 26 de noviembre... Después de esa fecha..."

El agente **no concatena respuestas** literalmente, sino que **razona** sobre ambas fuentes para generar una respuesta personalizada.


---

## Evidencias 

* **[Código ejecutado en Google Colab](https://colab.research.google.com/drive/1L_Hdf3ag5qmC-h0JEIkt67sLAxNCt5Tt?usp=sharing)** 