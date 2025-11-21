# 🤖 SmartBot AI: Arquitectura Inteligente de Soporte con RAG Híbrido y Respuestas Estructuradas

## Contexto

Los sistemas de soporte técnico tradicionales enfrentan el desafío de balancear escalabilidad con personalización. Los chatbots basados en reglas resultan rígidos y limitados, mientras que los sistemas puramente generativos tienden a alucinar información incorrecta. Este proyecto implementa un **chatbot híbrido de soporte técnico** que combina lo mejor de ambos mundos: recuperación de información fundamentada (RAG) desde una base de conocimiento local y búsqueda web complementaria para preguntas no cubiertas.

Se desarrolló un sistema de soporte para **SmartBot AI**, un producto ficticio de domótica, que responde consultas técnicas de usuarios integrando tres componentes clave: **vector store FAISS** con 8 FAQs oficiales, **módulo de decisión inteligente** que determina cuándo buscar información adicional en web, y **salida estructurada con Pydantic** que garantiza respuestas válidas con fuentes citadas y nivel de confianza explícito. El proyecto se ejecutó completamente en Google Colab, reutilizando la infraestructura de LangChain configurada en prácticas anteriores.

---

## 🎯 Objetivos

* **Construir corpus local** de FAQs indexado con embeddings de OpenAI y FAISS para recuperación semántica eficiente.
* **Implementar búsqueda web** como fallback cuando la información local es insuficiente o ausente.
* **Diseñar esquema Pydantic** para salidas estructuradas con respuesta, fuentes citadas (title+URL) y nivel de confianza.
* **Desarrollar lógica de decisión** que determine automáticamente cuándo recurrir a búsqueda web basándose en relevancia de documentos locales.
* **Integrar pipeline completo** de RAG híbrido con prompt engineering para priorizar información oficial sobre resultados web.
* **Validar sistema** con casos de prueba representativos que cubran preguntas con respuesta en FAQs, problemas técnicos específicos, y consultas fuera del dominio.
* **Analizar métricas** de confianza, longitud de respuestas y fuentes utilizadas para evaluar comportamiento del sistema.
* **Documentar arquitectura** y decisiones de diseño en README técnico completo.

---

## 📋 Actividades y Resultados

| Actividad | Descripción | Resultado Obtenido |
|-----------|-------------|-------------------|
| **1. Corpus local de FAQs** | Creación de 8 documentos sobre SmartBot AI (reinicio, conectividad, actualizaciones, compatibilidad, garantía, control de dispositivos). Splitting con RecursiveCharacterTextSplitter e indexación en FAISS. | Corpus indexado: **8 chunks de 500 caracteres**. Embeddings: OpenAI text-embedding-ada-002. Retriever configurado con k=3. |
| **2. Búsqueda web** | Instalación de DuckDuckGo Search como herramienta de búsqueda externa. Implementación de fallback simulado ante errores de conectividad. | Búsqueda web configurada con **MockWebSearch** por limitaciones del entorno Colab (error de importación ddgs). Sistema funcional con respuestas simuladas. |
| **3. Esquema Pydantic** | Diseño de clases `Source` (title, url) y `ChatbotResponse` (answer, sources, confidence) con validación automática. | Esquema validado con ejemplo: reinicio de SmartBot con fuente "FAQ local" y confidence="high". JSON bien formado generado correctamente. |
| **4. Lógica de decisión** | Implementación de función `should_search_web()` que evalúa si documentos locales son suficientes basándose en cantidad (threshold: 2 docs mínimo). | Lógica funcionando: **False** con 3 docs (info suficiente), **True** sin docs (necesita web). Umbral de 0.5 para scores de relevancia. |
| **5. Chatbot integrado** | Pipeline completo: retriever → decisión web → prompt con contexto combinado → LLM structured output. Prompt engineering con instrucciones de priorización de fuentes. | Primera pregunta ("¿Cómo reinicio?") respondida con **confidence=high**, fuente FAQ local, respuesta completa con pasos y tiempo estimado (30 seg). |
| **6. Casos de prueba** | Validación con 4 escenarios: FAQ cubierta, problema técnico, pregunta fuera de dominio (competidores), y términos de garantía. | **Caso 1:** Funcionamiento sin internet → high confidence, FAQ local. **Caso 2:** No enciende tras actualizar → high, troubleshooting correcto. **Caso 3:** Comparación con Alexa → **low confidence**, sin fuentes (info insuficiente). **Caso 4:** Garantía 2 años → high, FAQ local. |
| **7. Análisis de métricas** | Cálculo de distribución de confianza, fuentes utilizadas y longitud promedio de respuestas en los 4 casos. | **Distribución:** 3 casos high, 1 low. **Fuentes:** 3/4 usaron FAQ local, 1 sin fuentes. **Longitud promedio:** 288 caracteres por respuesta. |
| **8. Documentación técnica** | Generación de README.md con descripción de arquitectura, datos, parámetros, lógica de decisión, limitaciones y mejoras futuras. | README completo con diagrama de flujo, tabla de asignación de confianza, ejemplos de uso y 5 mejoras propuestas (memoria conversacional, fine-tuning, clasificación de intención, fallback humano, cache). |

---

## 🔬 Desarrollo

### 🗂️ Parte 1: Construcción del corpus local

Se creó una base de conocimiento con 8 FAQs oficiales sobre SmartBot AI cubriendo temas fundamentales de soporte: reinicio del sistema, funcionamiento offline, actualizaciones de firmware, troubleshooting de problemas comunes, compatibilidad con Apple, cambio de idioma, términos de garantía y control de dispositivos inteligentes.

Los documentos fueron formateados en markdown con estructura pregunta-respuesta clara. Se utilizó `RecursiveCharacterTextSplitter` con chunk_size=500 y chunk_overlap=50 para mantener coherencia contextual entre fragmentos. El vector store FAISS se construyó con `OpenAIEmbeddings` (modelo text-embedding-ada-002), generando 8 chunks indexados semánticamente.

El retriever se configuró con k=3 para recuperar los 3 documentos más relevantes ante cada consulta. Esta elección balancea cobertura de información (suficientes documentos para contexto completo) con precisión (evitar ruido de documentos poco relevantes).

**Resultado clave:** El corpus de 8 chunks permite responder la mayoría de consultas básicas sin necesidad de búsqueda web, como demostró el análisis posterior donde 3 de 4 casos se resolvieron completamente con información local.

---

### 🌐 Parte 2: Integración de búsqueda web

Se intentó integrar DuckDuckGo Search como herramienta de búsqueda externa mediante `langchain_community.tools.DuckDuckGoSearchRun`. Sin embargo, el entorno Colab presentó error de importación del módulo `ddgs`, común en ambientes con restricciones de red o versiones incompatibles de dependencias.

Como solución pragmática, se implementó `MockWebSearch`, una clase que simula búsqueda web devolviendo mensajes informativos sin realizar requests reales. Esta aproximación permite completar el ejercicio académico manteniendo la lógica de decisión y flujo del sistema intactos.

---

### 📋 Parte 3: Diseño de esquema estructurado

Se definieron dos clases Pydantic para garantizar respuestas bien formadas:

**`Source`:** Representa una fuente citada con campos `title` (descripción de la fuente) y `url` (referencia o "FAQ local" para documentos internos). Esta estructura permite auditoría de respuestas y trazabilidad de información.

**`ChatbotResponse`:** Esquema principal con tres campos:

- `answer` (str): Respuesta completa y profesional al usuario.
- `sources` (List[Source]): Lista de todas las fuentes consultadas, permitiendo citación múltiple.
- `confidence` (Literal["low", "medium", "high"]): Nivel de confianza explícito basado en calidad y origen de información.

El uso de `Field()` con descriptions permite que el LLM entienda semánticamente qué se espera en cada campo. El tipo `Literal` en confidence restringe valores a opciones válidas, evitando respuestas como "muy alta" o "moderada".

**Validación exitosa:** El ejemplo de prueba generó JSON perfectamente formado con respuesta sobre reinicio, fuente FAQ local, y confidence="high". El sistema de types garantiza que nunca se generen respuestas sin estructura o con campos faltantes.

---

### 🧠 Parte 4: Lógica de decisión inteligente

La función `should_search_web()` implementa el núcleo de la decisión híbrida:

**Regla 1:** Si no hay documentos locales (lista vacía) → buscar web inmediatamente.

**Regla 2:** Si hay metadata con scores de similitud, calcular promedio y comparar contra threshold (0.5 default). Scores bajos indican baja relevancia → buscar web.

**Regla 3:** Como heurística simple, asumir que si se recuperaron menos de 2 documentos, la información es insuficiente.

Esta lógica evita llamadas innecesarias a búsqueda web (costosas en latencia y $) cuando las FAQs ya tienen respuesta completa, pero garantiza que preguntas fuera del dominio activen búsqueda complementaria.

**Pruebas de validación:**

- `should_search_web([doc1, doc2, doc3])` → **False**: Suficientes docs locales.
- `should_search_web([])` → **True**: Sin información local.

El threshold de 0.5 fue elegido empíricamente; en producción se ajustaría mediante A/B testing evaluando satisfacción del usuario vs costo de búsqueda web.

---

### 🤖 Parte 5: Pipeline integrado del chatbot

El corazón del sistema combina todos los componentes previos en una función `smartbot_support()`:

**Flujo de ejecución:**

1. **Recuperación local:** Invocar retriever con pregunta del usuario, obtener top-3 documentos más similares.

2. **Decisión de búsqueda:** Evaluar si docs locales son suficientes con `should_search_web()`.

3. **Búsqueda complementaria (condicional):** Si necesario, ejecutar `web_search.run()` limitando resultados a 800 caracteres para no saturar contexto del LLM.

4. **Construcción de prompt:** Usar `ChatPromptTemplate` con dos mensajes:

   - **System:** Define rol ("asistente de soporte técnico"), instrucciones de priorización (FAQs > web), y criterios de asignación de confidence.
   - **Human:** Inyecta contexto local + contexto web (si existe) + pregunta del usuario.

5. **Generación estructurada:** Invocar chain (prompt | llm_structured) que garantiza respuesta con esquema `ChatbotResponse` válido.

**Prompt engineering destacado:**

- "Usa SOLO la información del contexto proporcionado" → evita alucinaciones.
- "Si tienes información de FAQs locales, priorízala" → sesgo correcto hacia fuente oficial.
- "Si la información es insuficiente, dilo honestamente" → admitir incertidumbre en lugar de inventar.
- Criterios explícitos de confidence → consistency en asignación de niveles.

**Primera ejecución exitosa:** Pregunta "¿Cómo reinicio mi SmartBot?" recuperó FAQ de reinicio, determinó que info local era suficiente (no buscó web), generó respuesta con pasos detallados ("Configuración > Sistema > Reiniciar"), comando de voz alternativo, y tiempo estimado. Confidence=high, fuente=FAQ local. Respuesta profesional y completa.

---

### 🧪 Parte 6: Validación con casos de prueba

Se ejecutaron 4 escenarios diseñados para cubrir espectro completo de comportamiento:

**CASO 1 - Pregunta cubierta en FAQs:** "¿SmartBot funciona sin internet?"

- **Resultado:** Respuesta correcta identificando funcionalidad offline (alarmas, temporizadores) vs online (actualizaciones, búsquedas).
- **Confidence:** high (100% basado en FAQ oficial).
- **Fuente:** FAQ local.
- **Análisis:** Sistema funcionando óptimamente; no necesitó web.

**CASO 2 - Problema técnico específico:** "SmartBot no enciende después de actualizar"

- **Resultado:** Troubleshooting paso a paso (verificar corriente, WiFi, reinicio forzado 10 seg, contacto soporte).
- **Confidence:** high (combinó FAQ de troubleshooting + actualizaciones).
- **Fuente:** FAQ local.
- **Análisis:** El retriever recuperó docs relevantes de dos FAQs distintas; LLM sintetizó solución coherente. Esto demuestra capacidad de combinar múltiples documentos.

**CASO 3 - Pregunta fuera del dominio:** "¿SmartBot es mejor que Alexa?"

- **Resultado:** Admisión honesta de falta de información comparativa. Mencionó características de SmartBot (compatibilidad, ecosistema Apple) sin hacer afirmaciones sobre Alexa.
- **Confidence:** **low** (información indirecta e incompleta).
- **Fuentes:** Lista vacía (crítico: el sistema no inventó fuentes inexistentes).
- **Análisis:** Comportamiento esperado y deseable. El sistema reconoce límites de su conocimiento en lugar de alucinar comparaciones. En producción, este sería el trigger para escalar a agente humano.

**CASO 4 - Términos de garantía:** "¿Cuánto dura la garantía de SmartBot?"

- **Resultado:** Respuesta precisa (2 años contra defectos de fabricación, excluye uso indebido/accidentes).
- **Confidence:** high (FAQ específica de garantía).
- **Fuente:** FAQ local.
- **Análisis:** Extracción perfecta de información estructurada (duración, cobertura, exclusiones).

**Patrón observado:** El sistema tiene 75% de tasa de confianza alta (3/4 casos), lo cual es excelente para un corpus de solo 8 FAQs. El caso de baja confianza fue manejado correctamente mediante admisión de incertidumbre.

---

### 📊 Parte 7: Análisis cuantitativo de métricas

**Distribución de confianza:**

- **High:** 3 casos (75%) → Mayoría de consultas resueltas con información oficial.
- **Low:** 1 caso (25%) → Pregunta fuera de dominio reconocida apropiadamente.
- **Medium:** 0 casos → No se activó búsqueda web (por MockWebSearch), por lo que no hubo mezcla de fuentes.

**Fuentes utilizadas:**

- **"FAQ local":** 3 casos (Reinicio, Problema técnico, Garantía).
- **Sin fuentes:** 1 caso (Comparación competidores) → coherente con confidence=low.

**Longitud promedio de respuestas:** 288 caracteres.

- Respuestas concisas pero completas.
- Caso 2 (problema técnico) fue el más largo con lista numerada de pasos.
- Caso 3 (comparación) fue el más corto con admisión de falta de info.

**Insight clave:** El sistema mantiene consistencia entre confidence, presencia de fuentes, y completitud de respuestas. No hay casos anómalos donde confidence=high con respuesta vaga, o confidence=low con respuesta detallada.

---

### 📝 Parte 8: Documentación de arquitectura

Se generó README técnico completo (`README_SmartBot.md`) que incluye:

**Diagrama de flujo ASCII** mostrando pipeline: Usuario → Retriever → Decisión → LLM → Respuesta estructurada.

**Especificación de datos:** Corpus de 8 FAQs, temas cubiertos, formato markdown, indexación con FAISS.

**Lógica de decisión documentada:** Criterios exactos de cuándo se activa búsqueda web (< 2 docs, score < 0.5).

**Parámetros del modelo:** gpt-4o-mini, temperature=0, embeddings text-embedding-ada-002, k=3.

**Tabla de asignación de confidence:** Mapeo explícito de criterios (FAQs oficiales → high, mix FAQs+web → medium, solo web/escaso → low).

**Ejemplo de uso:** Snippet de código mostrando invocación y acceso a campos de respuesta.

**Limitaciones reconocidas:** 

- Búsqueda web limitada a 800 caracteres.
- Sistema stateless (sin memoria conversacional).
- Threshold fijo en lugar de dinámico.
- Latencia variable de DuckDuckGo.

Esta documentación permite que cualquier desarrollador entienda, replique y extienda el sistema sin necesidad de leer todo el código fuente.

---

## 💭 Reflexión

Este proyecto demuestra la potencia de **arquitecturas híbridas** en sistemas conversacionales de producción. La integración de RAG con búsqueda web representa un equilibrio pragmático entre fundamentación factual y cobertura comprehensiva.

Este desafío integrador forzó aplicación simultánea de múltiples conceptos: vector stores, RAG, structured output, prompt engineering, decisión condicional, evaluación. Es representativo de la complejidad real de sistemas LLM en producción, donde raramente un solo componente resuelve todo el problema.

La iteración rápida en Colab (ejecutar, observar resultados, ajustar, repetir) demostró la importancia de **observabilidad**: sin prints informativos (`"🔍 Buscando..."`, `"✅ Info local suficiente"`) habría sido imposible debuggear el flujo de decisión. En producción, esto se traduce a logging estructurado con niveles (DEBUG, INFO, WARN) y tracing distribuido (LangSmith, Phoenix).

Finalmente, el ejercicio de documentar en README.md no fue meramente burocrático: forzó clarificación de decisiones de diseño que estaban implícitas. El acto de explicar "por qué threshold=0.5" o "por qué k=3" genera reflexión que mejora el diseño. 

---

## 📊 Evidencias

* [Código ejecutado completo en Google Colab](https://colab.research.google.com/drive/1PU6sfXvtbNwPR2nA73gK2poh3XFkEO4r?usp=sharing)