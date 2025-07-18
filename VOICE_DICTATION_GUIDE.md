# 🎤 Guía de Dictado de Voz - Thinker AI Auxiliary Window

## 📋 Descripción General

El **Sistema de Dictado de Voz** permite convertir voz a texto utilizando múltiples motores de reconocimiento, integrándose perfectamente con la interfaz de chat de Thinker AI.

### ✨ **Características Principales**
- 🎙️ **Dictado instantáneo** con un clic
- 🔄 **Múltiples motores** de reconocimiento (Google, Whisper, Azure)
- 🌐 **Soporte en español** nativo
- 🔁 **Modo conversación continua** para diálogos fluidos
- 🎯 **Integración perfecta** con el chat existente
- 📊 **Indicadores visuales** de estado
- 🧪 **Sistema de pruebas** integrado

---

## 🚀 **Instalación y Configuración**

### **Paso 1: Instalar Dependencias**

```bash
# Instalar paquetes de Python necesarios
pip install SpeechRecognition>=3.10.0 pyaudio>=0.2.11

# O instalar todas las dependencias del proyecto
pip install -r requirements.txt
```

### **Paso 2: Configurar Permisos de Micrófono**

#### **Windows:**
1. Ve a **Configuración > Privacidad > Micrófono**
2. Habilita **"Permitir que las aplicaciones accedan al micrófono"**
3. Verifica que Python esté en la lista de aplicaciones permitidas

#### **macOS:**
1. Ve a **Preferencias del Sistema > Seguridad y Privacidad > Micrófono**
2. Agrega **Terminal** o **Python** a la lista de aplicaciones permitidas

#### **Linux:**
```bash
# Verificar que el micrófono esté detectado
arecord -l

# Instalar ALSA si es necesario
sudo apt-get install libasound2-dev
```

### **Paso 3: Configurar Motor de Reconocimiento**

Edita `src/config/config.py`:

```python
# Speech Recognition Configuration
SPEECH_ENGINE = "google"          # "google", "whisper", "azure"
SPEECH_LANGUAGE = "es-ES"         # Español de España
SPEECH_ENERGY_THRESHOLD = 300     # Sensibilidad del micrófono
SPEECH_PAUSE_THRESHOLD = 3.0      # Segundos de silencio para enviar respuesta automáticamente
```

---

## 🎯 **Cómo Usar el Dictado de Voz**

### **🎤 Dictado Básico**

1. **Abrir Thinker AI** y ir al chat principal
2. **Hacer clic en el botón 🎤** (botón rojo al lado del envío)
3. **Hablar claramente** cuando aparezca "🔴 Grabando..."
4. **Permanecer en silencio durante 3 segundos** para finalizar automáticamente
5. **El texto se enviará automáticamente** al asistente de IA
6. **Continuar hablando** para mantener la conversación fluida
7. **Hacer clic en ⏹️** (botón verde) para finalizar el modo conversación

> **¡Nuevo!** El sistema ahora funciona en modo conversación continua. Habla, espera la respuesta, y sigue hablando sin necesidad de hacer clic en el botón de micrófono cada vez.

### **🔄 Estados del Botón de Voz**

| **Estado** | **Icono** | **Color** | **Descripción** |
|------------|-----------|-----------|-----------------|
| Listo | 🎤 | Rojo | Preparado para grabar |
| Conversación Continua | ⏹️ | Verde | Modo conversación activo |
| Escuchando | 🔴 | Rojo | Detectando voz activamente |
| Procesando | 🎤 | Azul | Reconociendo el texto |

### **🔁 Modo Conversación Continua**

El nuevo modo de conversación continua permite mantener un diálogo fluido con el asistente:

1. **Activación**: Haz clic en el botón 🎤 para iniciar el modo conversación
2. **Indicador**: El botón cambia a ⏹️ (verde) indicando que el modo está activo
3. **Uso**: 
   - Habla cuando quieras hacer una pregunta
   - Espera 3 segundos en silencio para que se envíe automáticamente
   - El asistente responderá a tu pregunta
   - Continúa hablando para hacer la siguiente pregunta
4. **Desactivación**: Haz clic en el botón ⏹️ para finalizar el modo conversación

> **Ventaja**: No necesitas hacer clic en el botón de micrófono cada vez que quieras hablar, permitiendo una conversación más natural y fluida.

### **⚙️ Menú de Sistema - Opciones de Voz**

Accede a través del menú **⚙️ > 🎤 Test Voz**:
- **Probar reconocimiento** - Test completo del sistema
- **Estado del micrófono** - Verificar hardware disponible
- **Estadísticas de uso** - Métricas de rendimiento

---

## 🧪 **Sistema de Pruebas**

### **Test Automático**
```
Menu ⚙️ > 🎤 Test Voz
```
- **Detecta micrófonos** disponibles
- **Prueba el motor** de reconocimiento
- **Verifica la calidad** del audio
- **Muestra configuración** actual

### **Test Manual**
1. Hacer clic en **🎤 Test Voz**
2. Decir una frase cuando se solicite
3. Verificar que el texto se reconozca correctamente
4. Revisar métricas de tiempo y precisión

### **Diagnóstico de Problemas**

#### **🔴 "Micrófono No Disponible"**
```
✅ Soluciones:
• Verificar que el micrófono esté conectado
• Revisar permisos del sistema
• Reiniciar la aplicación
• Comprobar drivers de audio
```

#### **🔴 "No se reconoce la voz"**
```
✅ Soluciones:
• Hablar más alto y claro
• Reducir ruido de fondo
• Ajustar SPEECH_ENERGY_THRESHOLD
• Cambiar motor de reconocimiento
```

#### **🔴 "Error de dependencias"**
```
✅ Soluciones:
• pip install SpeechRecognition pyaudio
• Verificar versión de Python (3.8+)
• Instalar Visual Studio Build Tools (Windows)
```

---

## ⚙️ **Configuración Avanzada**

### **📱 Motores de Reconocimiento**

#### **🌐 Google Speech Recognition (Recomendado)**
```python
SPEECH_ENGINE = "google"
# ✅ Ventajas: Rápido, preciso, gratis hasta cierto límite
# ❌ Desventajas: Requiere internet
```

#### **🤖 OpenAI Whisper (Offline)**
```python
SPEECH_ENGINE = "whisper"
# ✅ Ventajas: Funciona sin internet, muy preciso
# ❌ Desventajas: Requiere instalación adicional
```

#### **☁️ Azure Speech Services**
```python
SPEECH_ENGINE = "azure"
AZURE_SPEECH_KEY = "tu-api-key"
AZURE_SPEECH_REGION = "westus2"
# ✅ Ventajas: Empresarial, múltiples idiomas
# ❌ Desventajas: Requiere suscripción paga
```

### **🎚️ Ajustes de Sensibilidad**

```python
# Micrófono muy sensible (ambientes silenciosos)
SPEECH_ENERGY_THRESHOLD = 100

# Micrófono moderado (uso normal)
SPEECH_ENERGY_THRESHOLD = 300

# Micrófono poco sensible (ambientes ruidosos)
SPEECH_ENERGY_THRESHOLD = 1000
```

### **⏱️ Configuración de Tiempos**

```python
SPEECH_TIMEOUT = 5.0              # Tiempo máximo esperando voz
SPEECH_PHRASE_TIME_LIMIT = 10.0   # Tiempo máximo de frase completa
SPEECH_PAUSE_THRESHOLD = 3.0      # Segundos de silencio para enviar respuesta automáticamente
```

---

## 🔧 **Integración con Chat**

### **📝 Flujo de Trabajo Típico**

#### **Modo Estándar (Dictado Único):**
1. **Usuario:** Hacer clic en 🎤
2. **Sistema:** Iniciar grabación automáticamente
3. **Usuario:** Hablar mensaje ("Explícame qué es Python")
4. **Sistema:** Reconocer y mostrar texto
5. **Usuario:** Revisar/editar si es necesario
6. **Usuario:** Presionar Enter para enviar a IA
7. **IA:** Procesar y responder como siempre

#### **Modo Conversación Continua (Nuevo):**
1. **Usuario:** Hacer clic en 🎤 (cambia a ⏹️)
2. **Sistema:** Activar modo conversación continua
3. **Usuario:** Hablar mensaje ("Explícame qué es Python")
4. **Sistema:** Detectar silencio de 3 segundos y enviar automáticamente
5. **IA:** Procesar y responder
6. **Usuario:** Hablar siguiente pregunta sin hacer clic ("¿Y para qué sirve?")
7. **Sistema:** Detectar silencio y enviar automáticamente
8. **IA:** Procesar y responder
9. **Usuario:** Hacer clic en ⏹️ para finalizar cuando termine la conversación

### **🎯 Consejos para Mejor Reconocimiento**

#### **📢 Técnica de Habla:**
- **Hablar claramente** y con ritmo normal
- **Evitar muletillas** ("eh", "umm", "este")
- **Pausar entre frases** para mejor reconocimiento
- **Usar puntuación verbal** ("punto", "coma", "signo de pregunta")
- **Permanecer en silencio 3 segundos** al terminar para envío automático

#### **🎧 Configuración de Audio:**
- **Usar micrófono dedicado** mejor que el integrado
- **Ambiente silencioso** sin ecos
- **Distancia correcta** (15-30 cm del micrófono)
- **Hablar hacia el micrófono** directamente

---

## 📊 **Monitoreo y Estadísticas**

### **🔍 Estado del Sistema**
Accede a través de **Menu ⚙️ > 📊 Estado**:

```
🎤 Speech Recognition Status:
  Available: True
  Engine: google
  Language: es-ES
  Microphone: Available
  Microphones: 2
  Last Recognition: 2025-01-15T10:30:45
```

### **📈 Métricas de Rendimiento**
- **Tiempo de reconocimiento** típico: 1-3 segundos
- **Precisión esperada** en español: 85-95%
- **Longitud óptima** de frase: 5-20 palabras

---

## 🚨 **Solución de Problemas**

### **🔴 Problemas Comunes**

#### **Error: "No module named 'speech_recognition'"**
```bash
pip install SpeechRecognition
```

#### **Error: "No module named 'pyaudio'"**
```bash
# Windows
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install python3-pyaudio
```

#### **Error: "Microsoft Visual C++ 14.0 is required"**
```bash
# Windows - Instalar Visual Studio Build Tools
# O usar wheels precompilados:
pip install pyaudio --only-binary=all
```

### **⚡ Optimización de Rendimiento**

#### **🏃‍♂️ Reconocimiento Más Rápido:**
```python
SPEECH_TIMEOUT = 3.0           # Reducir timeout
SPEECH_PHRASE_TIME_LIMIT = 5.0 # Frases más cortas
```

#### **🎯 Mayor Precisión:**
```python
SPEECH_ENERGY_THRESHOLD = 500  # Aumentar umbral
SPEECH_PAUSE_THRESHOLD = 4.0   # Silencio más largo para detección automática
```

#### **⚡ Detección de Silencio:**
```python
SPEECH_PAUSE_THRESHOLD = 2.0   # Detección más rápida (2 segundos)
SPEECH_PAUSE_THRESHOLD = 3.0   # Valor predeterminado (3 segundos)
SPEECH_PAUSE_THRESHOLD = 4.0   # Detección más lenta (4 segundos)
```

---

## 🔮 **Roadmap Futuro**

### **🚀 Funcionalidades Planificadas**

#### **📚 Versión 1.1** ✅
- **Comandos de voz** directos ("enviar mensaje", "limpiar chat")
- ✅ **Grabación continua** sin necesidad de clicks (¡Implementado!)
- **Múltiples idiomas** simultáneos

#### **🤖 Versión 1.2**
- **Integración con IA local** (Whisper offline)
- **Reconocimiento de emociones** en la voz
- **Síntesis de voz** para respuestas del AI

#### **🌐 Versión 1.3**
- **Transcripción en tiempo real** durante reuniones
- **Traducción automática** de voz
- **Análisis de sentimientos** vocal

---

## 🔗 **Referencias y Recursos**

### **📚 Documentación Técnica**
- [SpeechRecognition Documentation](https://pypi.org/project/SpeechRecognition/)
- [PyAudio Documentation](https://pypi.org/project/PyAudio/)
- [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)

### **🛠️ Herramientas de Desarrollo**
- [Audacity](https://www.audacityteam.org/) - Editor de audio para pruebas
- [VoiceMeeter](https://vb-audio.com/Voicemeeter/) - Mezclador virtual de audio

### **🎯 Mejores Prácticas**
- **Calibrar** el micrófono antes del primer uso
- **Probar diferentes motores** según tu caso de uso
- **Mantener actualizado** SpeechRecognition regularmente

---

## 🤝 **Soporte y Contribuciones**

### **❓ Obtener Ayuda**
- **Logs detallados** en `logs/thinker_aux.log`
- **Test automático** con Menu ⚙️ > 🎤 Test Voz
- **Estado del sistema** con Menu ⚙️ > 📊 Estado

### **🔧 Contribuir Mejoras**
- **Reportar bugs** con logs completos
- **Sugerir mejoras** de reconocimiento
- **Compartir configuraciones** optimizadas para diferentes idiomas

---

*Documentación del Dictado de Voz para Thinker AI Auxiliary Window v1.0.0*  
*Última actualización: Enero 2025* 🎤✨ 
