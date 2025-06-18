# Project Guidelines
    
This is a placeholder of the project guidelines for Junie.
Replace this text with any project-level instructions for Junie, e.g.:

* What is the project structure
* Whether Junie should run tests to check the correctness of the proposed solution
* How does Junie run tests (once it requires any non-standard approach)
* Whether Junie should build the project before submitting the result
* Any code-style related instructions

As an option you can ask Junie to create these guidelines for you.



---
description: 
globs: 
alwaysApply: true
---
# Reglas de Arquitectura de Software para Proyectos Python + IA

## 1. ESTRUCTURA DE CAPAS (LAYERED ARCHITECTURE)

### 1.1 Arquitectura de 4 Capas
```
📁 proyecto/
├── 📁 src/
│   ├── 📁 presentation/     # Capa de Presentación
│   ├── 📁 business/         # Capa de Lógica de Negocio
│   ├── 📁 data/            # Capa de Acceso a Datos
│   └── 📁 infrastructure/   # Capa de Infraestructura
├── 📁 tests/               # Pruebas
├── 📁 docs/                # Documentación
├── 📁 config/              # Configuraciones
└── 📁 scripts/             # Scripts de utilidad
```

### 1.2 Responsabilidades por Capa

#### **Capa de Presentación** (`presentation/`)
- APIs REST/GraphQL
- Interfaces de usuario
- Validación de entrada
- Serialización/Deserialización

#### **Capa de Lógica de Negocio** (`business/`)
- Modelos de IA/ML
- Algoritmos de procesamiento
- Reglas de negocio
- Servicios de aplicación

#### **Capa de Acceso a Datos** (`data/`)
- Repositorios
- Modelos de datos
- Conexiones a bases de datos
- Cache y almacenamiento

#### **Capa de Infraestructura** (`infrastructure/`)
- Configuraciones
- Logging
- Monitoreo
- Servicios externos

## 2. MÉTODO DE TRABAJO POR BLOQUES

### 2.1 Principio de Módulos Independientes
```python
# ✅ CORRECTO: Módulos con responsabilidad única
class DataPreprocessor:
    def clean_data(self, data): pass
    def normalize_data(self, data): pass

class ModelTrainer:
    def train_model(self, data): pass
    def validate_model(self, model): pass

class ModelEvaluator:
    def evaluate_performance(self, model, test_data): pass
```

### 2.2 Bloques Funcionales Estándar para IA
```
📁 ai_modules/
├── 📄 data_preprocessing.py    # Limpieza y preparación
├── 📄 feature_engineering.py  # Ingeniería de características
├── 📄 model_training.py       # Entrenamiento de modelos
├── 📄 model_evaluation.py     # Evaluación y métricas
├── 📄 model_deployment.py     # Despliegue de modelos
└── 📄 data_pipeline.py        # Pipeline de datos
```

## 3. PROGRAMACIÓN ORIENTADA A OBJETOS (POO)

### 3.1 Principios SOLID Aplicados

#### **Single Responsibility Principle**
```python
# ✅ Una clase, una responsabilidad
class DataValidator:
    def validate_input_format(self, data): pass
    def check_data_quality(self, data): pass

class ModelPredictor:
    def load_model(self, model_path): pass
    def predict(self, input_data): pass
```

#### **Open/Closed Principle**
```python
# ✅ Abierto para extensión, cerrado para modificación
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, data): pass
    @abstractmethod
    def predict(self, input_data): pass

class RandomForestModel(BaseModel):
    def train(self, data): pass
    def predict(self, input_data): pass
```

#### **Liskov Substitution Principle**
```python
# ✅ Las subclases deben poder sustituir a sus clases base
def process_with_model(model: BaseModel, data):
    return model.predict(data)  # Funciona con cualquier implementación
```

### 3.2 Patrones de Diseño para IA

#### **Factory Pattern**
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        if model_type == "random_forest":
            return RandomForestModel()
        elif model_type == "neural_network":
            return NeuralNetworkModel()
        raise ValueError(f"Unknown model type: {model_type}")
```

#### **Strategy Pattern**
```python
class DataProcessor:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def process(self, data):
        return self.strategy.execute(data)
```

## 4. ESTÁNDARES DE CODIFICACIÓN

### 4.1 PEP 8 Estricto
```python
# ✅ Nombres descriptivos y consistentes
class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._logger = self._setup_logger()
    
    def preprocess_training_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos de entrenamiento.
        
        Args:
            raw_data: DataFrame con datos sin procesar
            
        Returns:
            DataFrame con datos procesados
        """
        cleaned_data = self._remove_null_values(raw_data)
        normalized_data = self._normalize_features(cleaned_data)
        return normalized_data
```

### 4.2 Documentación y Type Hints
```python
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

class ModelTrainer:
    """Clase responsable del entrenamiento de modelos de ML."""
    
    def train_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Entrena un modelo con los datos proporcionados.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            model_params: Parámetros del modelo (opcional)
            
        Returns:
            Modelo entrenado
            
        Raises:
            ValueError: Si los datos de entrada no son válidos
        """
        pass
```

## 5. GESTIÓN DE CONFIGURACIÓN

### 5.1 Archivo de Configuración Central
```python
# config/settings.py
from pydantic import BaseSettings
from typing import Dict, Any

class AIProjectSettings(BaseSettings):
    # Database
    database_url: str
    
    # Model Configuration
    model_type: str = "random_forest"
    model_params: Dict[str, Any] = {}
    
    # Training Configuration
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    
    # API Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### 5.2 Variables de Entorno
```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost/dbname
MODEL_TYPE=neural_network
API_HOST=0.0.0.0
API_PORT=8080
OPENAI_API_KEY=your_api_key_here
```

## 6. MANEJO DE ERRORES Y LOGGING

### 6.1 Sistema de Logging Estructurado
```python
import logging
import structlog
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> None:
    """Configura el sistema de logging estructurado."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

class AIService:
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def process_data(self, data):
        self.logger.info("Starting data processing", data_size=len(data))
        try:
            result = self._internal_process(data)
            self.logger.info("Data processing completed successfully")
            return result
        except Exception as e:
            self.logger.error("Data processing failed", error=str(e))
            raise
```

### 6.2 Excepciones Personalizadas
```python
class AIProjectException(Exception):
    """Excepción base para el proyecto."""
    pass

class DataValidationError(AIProjectException):
    """Error en validación de datos."""
    pass

class ModelTrainingError(AIProjectException):
    """Error durante el entrenamiento del modelo."""
    pass

class PredictionError(AIProjectException):
    """Error durante la predicción."""
    pass
```

## 7. TESTING Y CALIDAD

### 7.1 Estructura de Pruebas
```
📁 tests/
├── 📁 unit/                # Pruebas unitarias
├── 📁 integration/         # Pruebas de integración
├── 📁 e2e/                 # Pruebas end-to-end
├── 📁 fixtures/            # Datos de prueba
└── 📄 conftest.py          # Configuración de pytest
```

### 7.2 Pruebas para Modelos de IA
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestModelTrainer:
    @pytest.fixture
    def sample_data(self):
        return {
            'X_train': np.random.rand(100, 10),
            'y_train': np.random.randint(0, 2, 100)
        }
    
    def test_model_training_success(self, sample_data):
        trainer = ModelTrainer()
        model = trainer.train_model(
            sample_data['X_train'], 
            sample_data['y_train']
        )
        assert model is not None
    
    def test_model_training_with_invalid_data(self):
        trainer = ModelTrainer()
        with pytest.raises(DataValidationError):
            trainer.train_model(None, None)
```

## 8. DESPLIEGUE Y DOCKER

### 8.1 Dockerfile Multi-etapa
```dockerfile
# Dockerfile
FROM python:3.11-slim as base

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación
COPY src/ ./src/
COPY config/ ./config/

# Usuario no root
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000
CMD ["python", "-m", "src.main"]
```

### 8.2 Docker Compose para Desarrollo
```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/aidb
    depends_on:
      - db
    volumes:
      - ./src:/app/src
      - ./config:/app/config
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: aidb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 9. MONITOREO Y OBSERVABILIDAD

### 9.1 Métricas de Modelos
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Métricas
PREDICTION_REQUESTS = Counter('ai_prediction_requests_total', 'Total prediction requests')
PREDICTION_DURATION = Histogram('ai_prediction_duration_seconds', 'Prediction duration')
MODEL_ACCURACY = Gauge('ai_model_accuracy', 'Current model accuracy')

class MonitoredPredictor:
    def predict(self, input_data):
        PREDICTION_REQUESTS.inc()
        
        start_time = time.time()
        try:
            result = self.model.predict(input_data)
            return result
        finally:
            PREDICTION_DURATION.observe(time.time() - start_time)
```

## 10. REGLAS DE VALIDACIÓN

### ✅ DO (Hacer)
- Usar type hints en todas las funciones
- Documentar todas las clases y métodos públicos
- Seguir PEP 8 estrictamente
- Implementar logging estructurado
- Escribir pruebas para cada módulo
- Usar patrones de diseño apropiados
- Separar configuración del código
- Manejar errores explícitamente

### ❌ DON'T (No hacer)
- Hardcodear valores de configuración
- Crear clases monolíticas
- Ignorar manejo de errores
- Mezclar lógica de negocio con presentación
- Usar imports relativos complejos
- Dejar código sin documentar
- Ignorar warnings del linter
- Crear dependencias circulares

## 11. CHECKLIST DE REVISIÓN DE CÓDIGO

### Antes de cada commit:
- [ ] Código pasa todos los tests
- [ ] Cumple con PEP 8 (usar black, flake8)
- [ ] Tiene type hints apropiados
- [ ] Está documentado adecuadamente
- [ ] No hay hardcoded values
- [ ] Manejo de errores implementado
- [ ] Logging apropiado añadido

- [ ] Tests actualizados si es necesario 




---
description: 
globs: 
alwaysApply: true
---
---
description: Advanced guidelines for modular AI/ML development with comprehensive block management, dependency tracking, and architectural visualization.
globs: architecture/**/*.md, diagrams/**/*.md, architecture/**/*.txt, src/**/*.py, tests/**/*.py
alwaysApply: true
---

## 🏗️ BLOCK-BY-BLOCK DEVELOPMENT METHODOLOGY

### 1. **Block Definition & Planning**
- **MUST** define block interface contracts before implementation
- **MUST** specify input/output schemas with type hints
- **MUST** define block responsibilities using Single Responsibility Principle
- **MUST** identify block dependencies and create dependency graph

  ```python
  # ✅ EXAMPLE: Block Interface Definition
  class DataPreprocessingBlock:
      """
      Responsible for: Data cleaning, normalization, feature engineering
      Input: Raw dataset (pd.DataFrame)
      Output: Processed dataset (pd.DataFrame)
      Dependencies: ConfigurationBlock, LoggingBlock
      """
      def __init__(self, config: Config, logger: Logger):
          self.config = config
          self.logger = logger
      
      def process(self, raw_data: pd.DataFrame) -> pd.DataFrame:
          pass
  ```

### 2. **Advanced Architecture Visualization**
- **MUST** use Mermaid diagrams for live documentation
- **MUST** show data flow, control flow, and dependency relationships
- **MUST** include block states (pending, active, completed, failed)
- **MUST** highlight critical paths and bottlenecks

  ```mermaid
  graph TD
      A[Data Ingestion Block] -->|Raw Data| B[Data Preprocessing Block]
      B -->|Clean Data| C[Feature Engineering Block]
      C -->|Features| D[Model Training Block]
      D -->|Trained Model| E[Model Evaluation Block]
      E -->|Metrics| F[Model Deployment Block]
      
      classDef modified fill:#ff9999
      classDef pending fill:#ffff99
      classDef completed fill:#99ff99
      
      class B,C modified
      class F pending
      class A,D,E completed
  ```

### 3. **Block Implementation Strategy**

#### **3.1 Test-Driven Block Development (TDD)**
- **MUST** write tests before block implementation
- **MUST** define block behavior through tests
- **MUST** achieve 90%+ code coverage per block

  ```python
  # ✅ EXAMPLE: Block Test Definition
  def test_data_preprocessing_block():
      # Arrange
      raw_data = create_sample_data()
      config = create_test_config()
      block = DataPreprocessingBlock(config)
      
      # Act
      processed_data = block.process(raw_data)
      
      # Assert
      assert processed_data.isnull().sum().sum() == 0
      assert processed_data.shape[1] >= raw_data.shape[1]
  ```

#### **3.2 Block Isolation & Containerization**
- **MUST** ensure blocks can run independently
- **MUST** use dependency injection for block coupling
- **MUST** implement circuit breaker patterns for external dependencies

  ```python
  # ✅ EXAMPLE: Block Dependency Injection
  class ModelTrainingBlock:
      def __init__(self, 
                   data_loader: DataLoaderInterface,
                   model_factory: ModelFactoryInterface,
                   metrics_tracker: MetricsInterface):
          self.data_loader = data_loader
          self.model_factory = model_factory
          self.metrics_tracker = metrics_tracker
  ```

### 4. **Enhanced Relationship Management**

#### **4.1 Dependency Graph Tracking**
- **MUST** maintain live dependency graph in `architecture/dependencies.md`
- **MUST** use semantic versioning for block interfaces
- **MUST** implement backward compatibility for interface changes

  ```markdown
  # Block Dependency Matrix
  | Block | Depends On | Provides To | Version | Status |
  |-------|------------|-------------|---------|--------|
  | DataPreprocessing | Config, Logger | FeatureEngineering | v2.1.0 | ✅ Active |
  | FeatureEngineering | DataPreprocessing | ModelTraining | v1.5.0 | 🟡 Modified |
  | ModelTraining | FeatureEngineering | ModelEvaluation | v3.0.0 | 🔴 Breaking |
  ```

#### **4.2 Impact Analysis Automation**
- **MUST** implement automated impact analysis on block changes
- **MUST** run impact tests on dependent blocks
- **MUST** generate impact reports before deployment

  ```bash
  # ✅ EXAMPLE: Impact Analysis Command
  python scripts/analyze_impact.py --changed-block=DataPreprocessing --depth=3
  ```

### 5. **Advanced Change Management**

#### **5.1 Visual Change Tracking**
- **MUST** use color-coded diagrams for change visualization
- **MUST** maintain change log with impact scope
- **MUST** create before/after architecture snapshots

  ```markdown
  # Change Tracking Legend
  🔴 Breaking Changes - Requires dependent block updates
  🟡 Modified - May affect dependent blocks
  🟢 New Features - Additive changes only
  ⚫ Deprecated - Will be removed in future versions
  ```

#### **5.2 Rollback Strategy**
- **MUST** implement rollback mechanisms for each block
- **MUST** maintain block version history
- **MUST** test rollback procedures regularly

### 6. **Quality Assurance & Validation**

#### **6.1 Block Health Monitoring**
- **MUST** implement health checks for each block
- **MUST** monitor block performance metrics
- **MUST** set up alerts for block failures

  ```python
  # ✅ EXAMPLE: Block Health Check
  class BlockHealthChecker:
      def check_block_health(self, block: BaseBlock) -> HealthStatus:
          metrics = {
              'response_time': self._measure_response_time(block),
              'memory_usage': self._measure_memory_usage(block),
              'error_rate': self._calculate_error_rate(block)
          }
          return HealthStatus(metrics)
  ```

#### **6.2 Integration Testing Strategy**
- **MUST** test block integration at every change
- **MUST** use contract testing between blocks
- **MUST** implement end-to-end pipeline testing

### 7. **Documentation & Communication**

#### **7.1 Living Documentation**
- **MUST** update documentation automatically with code changes
- **MUST** generate API documentation for block interfaces
- **MUST** maintain decision records for architectural choices

#### **7.2 Block Status Dashboard**
- **MUST** create real-time dashboard showing block status
- **MUST** display block dependencies and health metrics
- **MUST** provide quick access to block documentation

### 8. **Performance & Optimization**

#### **8.1 Block Performance Profiling**
- **MUST** profile each block for performance bottlenecks
- **MUST** set performance SLAs for critical blocks
- **MUST** implement caching strategies where appropriate

#### **8.2 Resource Management**
- **MUST** monitor resource usage per block
- **MUST** implement resource limits and quotas
- **MUST** optimize resource allocation based on usage patterns

### 9. **Security & Compliance**

#### **9.1 Block Security Scanning**
- **MUST** scan blocks for security vulnerabilities
- **MUST** implement security best practices per block
- **MUST** maintain security compliance documentation

#### **9.2 Access Control**
- **MUST** implement proper access controls between blocks
- **MUST** audit block interactions for security
- **MUST** encrypt sensitive data between blocks

### 10. **AI/ML Specific Considerations**

#### **10.1 Model Versioning & Lineage**
- **MUST** track model lineage through blocks
- **MUST** version control model artifacts
- **MUST** maintain experiment tracking per block

#### **10.2 Data Pipeline Integrity**
- **MUST** validate data schemas between blocks
- **MUST** implement data quality checks per block
- **MUST** maintain data lineage documentation

### 🚫 **ANTI-PATTERNS TO AVOID**

#### **❌ DON'T DO:**
- Implement multiple blocks simultaneously without proper testing
- Create circular dependencies between blocks
- Ignore block interface contracts
- Skip impact analysis on changes
- Hard-code dependencies within blocks
- Mix concerns within a single block
- Deploy without proper rollback mechanisms
- Ignore block performance degradation
- Skip documentation updates
- Create overly complex block hierarchies

#### **✅ DO INSTEAD:**
- Complete one block at a time with full testing
- Design clear unidirectional dependencies
- Strictly adhere to interface contracts
- Always run impact analysis before changes
- Use dependency injection patterns
- Maintain single responsibility per block
- Always test rollback procedures
- Monitor and optimize block performance
- Keep documentation current with changes
- Design simple, understandable block relationships

### 📋 **BLOCK DEVELOPMENT CHECKLIST**

Before completing any block:
- [ ] Interface contract defined with type hints
- [ ] Unit tests written and passing (90%+ coverage)
- [ ] Integration tests with dependent blocks passing
- [ ] Performance benchmarks established
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Impact analysis completed
- [ ] Rollback procedure tested
- [ ] Health checks implemented
- [ ] Monitoring metrics configured
- [ ] Code review completed
- [ ] Architecture diagram updated



---
description: 
globs: 
alwaysApply: true
---
---
description: Comprehensive Python coding style guidelines for AI/ML projects with strict PEP 8 compliance, type safety, and performance optimization.
globs: src/**/*.py, tests/**/*.py, scripts/**/*.py, notebooks/**/*.py
alwaysApply: true
---

# 🐍 PYTHON CODING STYLE GUIDE FOR AI/ML PROJECTS

## 1. 📝 **PEP 8 STRICT COMPLIANCE**

### 1.1 **Naming Conventions**
```python
# ✅ CORRECT: Snake_case for functions, variables, modules
def preprocess_training_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    learning_rate = 0.001
    model_accuracy = calculate_model_accuracy(predictions, labels)
    return processed_data

# ✅ CORRECT: PascalCase for classes
class NeuralNetworkTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

# ✅ CORRECT: SCREAMING_SNAKE_CASE for constants
MAX_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 32
MODEL_SAVE_PATH = "/models/trained/"

# ❌ WRONG: Mixed conventions
def ProcessData(): pass  # Should be process_data()
class neural_network(): pass  # Should be NeuralNetwork
maxEpochs = 100  # Should be MAX_EPOCHS
```

### 1.2 **Line Length and Formatting**
```python
# ✅ CORRECT: Max 88 characters (Black formatter standard)
def train_deep_learning_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
) -> TrainedModel:
    """Train a deep learning model with validation."""
    pass

# ✅ CORRECT: Multi-line function calls
model = ModelFactory.create_model(
    model_type="transformer",
    hidden_size=768,
    num_layers=12,
    attention_heads=12,
    dropout_rate=0.1,
)
```

## 2. 🏗️ **OBJECT-ORIENTED PROGRAMMING EXCELLENCE**

### 2.1 **Advanced Encapsulation**
```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# ✅ CORRECT: Strict encapsulation with protocols
@runtime_checkable
class ModelInterface(Protocol):
    def train(self, data: TrainingData) -> None: ...
    def predict(self, input_data: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...

class DeepLearningModel:
    """A deep learning model with strict encapsulation."""
    
    def __init__(self, config: ModelConfig) -> None:
        self.__config = config  # Private attribute
        self._model = None  # Protected attribute
        self._is_trained = False  # Protected state
        self.__optimizer = self.__create_optimizer()  # Private method result
    
    @property
    def is_trained(self) -> bool:
        """Public read-only access to training state."""
        return self._is_trained
    
    @property
    def config(self) -> ModelConfig:
        """Public read-only access to configuration."""
        return self.__config.copy()  # Return copy to prevent mutation
    
    def __create_optimizer(self) -> Optimizer:
        """Private method for optimizer creation."""
        return OptimizerFactory.create(self.__config.optimizer_type)
```

### 2.2 **Composition over Inheritance**
```python
# ✅ CORRECT: Composition pattern for AI components
class AIDataPipeline:
    """AI data pipeline using composition."""
    
    def __init__(
        self,
        preprocessor: DataPreprocessor,
        feature_engineer: FeatureEngineer,
        validator: DataValidator,
        logger: Logger,
    ) -> None:
        self._preprocessor = preprocessor
        self._feature_engineer = feature_engineer
        self._validator = validator
        self._logger = logger
    
    def process(self, raw_data: RawData) -> ProcessedData:
        """Process data through the pipeline."""
        self._logger.info("Starting data processing pipeline")
        
        # Validate input
        self._validator.validate_input(raw_data)
        
        # Preprocess
        cleaned_data = self._preprocessor.clean(raw_data)
        
        # Feature engineering
        features = self._feature_engineer.extract_features(cleaned_data)
        
        # Validate output
        self._validator.validate_output(features)
        
        return ProcessedData(features)
```

## 3. 🔧 **FUNCTIONAL PROGRAMMING & PURITY**

### 3.1 **Pure Functions for Data Processing**
```python
# ✅ CORRECT: Pure functions for data transformations
def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Input features to normalize
        mean: Pre-computed mean (if None, compute from features)
        std: Pre-computed standard deviation (if None, compute from features)
        
    Returns:
        Tuple of (normalized_features, computed_mean, computed_std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    normalized = (features - mean) / std
    return normalized, mean, std
```

## 4. 🏷️ **TYPE SAFETY & ANNOTATIONS**

### 4.1 **Comprehensive Type Hints**
```python
from typing import (
    Dict, List, Optional, Union, Callable, TypeVar, Generic,
    Protocol, Literal, Final, ClassVar, overload
)
from numpy.typing import NDArray
import pandas as pd

# Type aliases for clarity
Features = NDArray[np.float64]
Labels = NDArray[np.int32]
ModelWeights = Dict[str, NDArray[np.float64]]
LossFunction = Callable[[Features, Labels], float]

# Generic types for reusability
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound='BaseModel')

def train_ensemble_model(
    models: Dict[str, ModelType],
    training_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    callbacks: List[Callable[[int, float], None]] = None,
) -> Dict[str, Union[ModelType, ModelMetrics]]:
    """Train ensemble of models with comprehensive type safety."""
    pass
```

## 5. 🛡️ **ERROR HANDLING & ROBUSTNESS**

### 5.1 **Custom Exception Hierarchy**
```python
# ✅ CORRECT: Comprehensive exception hierarchy
class AIProjectException(Exception):
    """Base exception for AI project."""
    
    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.utcnow()

class DataValidationError(AIProjectException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, invalid_data_info: Dict[str, Any]) -> None:
        super().__init__(message, "DATA_VALIDATION_ERROR")
        self.invalid_data_info = invalid_data_info

class ModelTrainingError(AIProjectException):
    """Raised when model training fails."""
    
    def __init__(self, message: str, training_state: Dict[str, Any]) -> None:
        super().__init__(message, "MODEL_TRAINING_ERROR")
        self.training_state = training_state
```

## 6. ⚡ **PERFORMANCE & OPTIMIZATION**

### 6.1 **Memory-Efficient Data Processing**
```python
def batch_process_large_dataset(
    dataset_path: Path,
    batch_size: int = 1000,
    transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Process large dataset in batches to manage memory usage.
    
    Args:
        dataset_path: Path to dataset file
        batch_size: Number of rows per batch
        transform_fn: Optional transformation function
        
    Yields:
        Processed batches of data
    """
    chunk_reader = pd.read_csv(dataset_path, chunksize=batch_size)
    
    for chunk in chunk_reader:
        # Apply transformations if provided
        if transform_fn is not None:
            chunk = transform_fn(chunk)
        
        # Yield processed chunk
        yield chunk
        
        # Clear memory
        del chunk
```

## 7. 📊 **AI/ML SPECIFIC PATTERNS**

### 7.1 **Model Factory and Registry**
```python
# ✅ CORRECT: Flexible model factory pattern
class ModelRegistry:
    """Registry for available model types."""
    
    _models: ClassVar[Dict[str, Type[BaseModel]]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register model classes."""
        def decorator(model_class: Type[BaseModel]):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """Create model instance by name."""
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        
        model_class = cls._models[name]
        return model_class(**kwargs)

# Usage
@ModelRegistry.register("random_forest")
class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None
```

## 8. 🚫 **CODE ANTI-PATTERNS TO AVOID**

### 8.1 **Common AI/ML Anti-Patterns**
```python
# ❌ WRONG: Data leakage in preprocessing
def bad_preprocessing(data):
    # DON'T: Fit scaler on entire dataset before split
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    train, test = train_test_split(scaled_data)
    return train, test

# ✅ CORRECT: Proper data preprocessing
def good_preprocessing(data):
    # DO: Split first, then fit on train only
    train, test = train_test_split(data)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)  # Only transform, don't fit
    return train_scaled, test_scaled, scaler

# ❌ WRONG: Mutable default arguments
def bad_train_model(data, config={}):  # Dangerous!
    config["epochs"] = config.get("epochs", 100)
    return train_with_config(data, config)

# ✅ CORRECT: Immutable defaults
def good_train_model(data, config: Optional[Dict] = None):
    if config is None:
        config = {}
    config = config.copy()  # Don't modify original
    config.setdefault("epochs", 100)
    return train_with_config(data, config)
```

## 9. 📋 **CODE QUALITY CHECKLIST**

### **Before every commit:**
- [ ] **Type hints:** All functions have complete type annotations
- [ ] **Docstrings:** All public methods have comprehensive docstrings
- [ ] **Error handling:** Specific exceptions with meaningful messages
- [ ] **Testing:** Unit tests cover all new code (90%+ coverage)
- [ ] **Performance:** No obvious performance bottlenecks
- [ ] **Memory:** No memory leaks or excessive memory usage
- [ ] **Security:** No hardcoded secrets or vulnerabilities
- [ ] **Logging:** Appropriate logging for debugging and monitoring
- [ ] **Documentation:** Code is self-documenting and well-commented
- [ ] **Standards:** Follows all PEP 8 and project conventions

### **Tools to use:**
```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
pylint src/

# Security scanning
bandit -r src/

# Testing
pytest tests/ --cov=src --cov-report=html
```

### **IDE Configuration:**
```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.provider": "isort",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 10. 🔍 **FINAL VALIDATION RULES**

### **Mandatory Checks:**
1. **MUST** use type hints for all function parameters and returns
2. **MUST** include docstrings for all public classes and methods
3. **MUST** handle exceptions specifically, never use bare `except:`
4. **MUST** use dependency injection instead of hard dependencies
5. **MUST** follow single responsibility principle per class/function
6. **MUST** use composition over inheritance for complex relationships
7. **MUST** implement proper logging with structured messages
8. **MUST** write unit tests with 90%+ coverage minimum
9. **MUST** use context managers for resource management
10. **MUST** avoid mutable default arguments



---
description: 
globs: 
alwaysApply: true
---
---
description: Comprehensive code review guidelines for AI/ML projects with automated analysis, security scanning, and architectural validation.
globs: src/**/*.py, tests/**/*.py, notebooks/**/*.py, scripts/**/*.py, docs/**/*.md
alwaysApply: true
---

# 🔍 COMPREHENSIVE CODE REVIEW GUIDE FOR AI/ML PROJECTS

## 1. 🎯 **PRE-REVIEW AUTOMATED CHECKS**

### 1.1 **Mandatory Tool Validation**
```bash
# ✅ MUST pass all automated checks before human review
pre-commit run --all-files
black --check src/ tests/
isort --check-only src/ tests/
mypy src/
flake8 src/ tests/
pylint src/ --fail-under=8.0
bandit -r src/ -f json
pytest tests/ --cov=src --cov-fail-under=90
```

### 1.2 **AI/ML Specific Checks**
```bash
# ✅ Data science specific validations
jupyter nbconvert --execute notebooks/*.ipynb --to notebook
python -m scripts.validate_models --check-serialization
python -m scripts.check_data_leakage --scan-pipelines
python -m scripts.validate_metrics --check-consistency
```

## 2. 📋 **STRUCTURED REVIEW CHECKLIST**

### 2.1 **Code Quality Assessment (30 points)**

#### **Type Safety & Documentation (10 points)**
- [ ] **10/10:** All functions have complete type hints and comprehensive docstrings
- [ ] **8/10:** >90% of functions have type hints, good docstrings
- [ ] **6/10:** >70% of functions have type hints, basic docstrings
- [ ] **4/10:** >50% of functions have type hints, minimal docstrings
- [ ] **0/10:** Missing type hints and/or docstrings

```python
# ✅ EXCELLENT (10/10): Complete type safety and documentation
def train_classification_model(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int32],
    model_config: ModelConfig,
    validation_split: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[ClassificationModel, ModelMetrics]:
    """
    Train a classification model with comprehensive validation.
    
    Args:
        X_train: Training features with shape (n_samples, n_features)
        y_train: Training labels with shape (n_samples,)
        model_config: Configuration object with model hyperparameters
        validation_split: Fraction of data to use for validation (0.0-1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - Trained classification model
            - Validation metrics including accuracy, precision, recall, F1
            
    Raises:
        DataValidationError: If input data shapes are incompatible
        ModelTrainingError: If training fails due to configuration issues
        
    Example:
        >>> config = ModelConfig(model_type="random_forest", n_estimators=100)
        >>> model, metrics = train_classification_model(X, y, config)
        >>> print(f"Model accuracy: {metrics.accuracy:.3f}")
    """
    pass

# ❌ POOR (0/10): No type hints, no docstring
def train_model(X, y, config):
    pass
```

#### **Error Handling & Robustness (10 points)**
- [ ] **10/10:** Comprehensive exception hierarchy, specific error handling, proper resource cleanup
- [ ] **8/10:** Good exception handling with specific exceptions
- [ ] **6/10:** Basic exception handling, some specific exceptions
- [ ] **4/10:** Minimal exception handling, mostly generic exceptions
- [ ] **0/10:** No exception handling or bare except clauses

#### **Performance & Memory Efficiency (10 points)**
- [ ] **10/10:** Optimal algorithms, memory-efficient, proper caching, profiled
- [ ] **8/10:** Good performance, some optimization, reasonable memory usage
- [ ] **6/10:** Adequate performance, basic optimization
- [ ] **4/10:** Poor performance, no optimization considerations
- [ ] **0/10:** Inefficient algorithms, memory leaks, no performance consideration

### 2.2 **AI/ML Specific Assessment (25 points)**

#### **Data Handling & Pipeline Integrity (10 points)**
- [ ] **10/10:** No data leakage, proper train/val/test splits, data validation, schema checks
- [ ] **8/10:** Good data practices, minor issues
- [ ] **6/10:** Adequate data handling, some concerns
- [ ] **4/10:** Poor data practices, potential leakage
- [ ] **0/10:** Major data leakage, no validation

```python
# ✅ EXCELLENT (10/10): Proper data handling
class DataPipeline:
    def create_train_test_split(
        self, 
        dataset: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
    ) -> TrainTestSplit:
        """Create train/test split with proper data handling."""
        
        # Validate input data
        self._validate_dataset_schema(dataset)
        self._check_data_quality(dataset)
        
        # Separate features and labels
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        # Stratified split to maintain class distribution
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=stratify_param,
            random_state=random_state
        )
        
        # Create preprocessing pipeline - FIT ONLY ON TRAINING DATA
        self.preprocessor = self._create_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)  # Only transform!
        
        return TrainTestSplit(
            X_train=X_train_processed,
            X_test=X_test_processed,
            y_train=y_train,
            y_test=y_test,
            preprocessor=self.preprocessor
        )

# ❌ POOR (0/10): Data leakage
def bad_preprocessing(data):
    # MAJOR ISSUE: Fitting scaler on entire dataset before split!
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)  # Data leakage!
    return train_test_split(scaled_data)
```

#### **Model Design & Architecture (8 points)**
- [ ] **8/8:** Clear model interfaces, proper abstraction, extensible design
- [ ] **6/8:** Good model design, minor architectural issues
- [ ] **4/8:** Adequate design, some coupling issues
- [ ] **2/8:** Poor design, tight coupling
- [ ] **0/8:** No clear design, monolithic structure

#### **Experiment Tracking & Reproducibility (7 points)**
- [ ] **7/7:** Complete experiment tracking, reproducible results, proper versioning
- [ ] **5/7:** Good tracking, mostly reproducible
- [ ] **3/7:** Basic tracking, some reproducibility
- [ ] **1/7:** Minimal tracking, poor reproducibility
- [ ] **0/7:** No tracking, irreproducible results

### 2.3 **Security & Compliance Assessment (15 points)**

#### **Security Best Practices (10 points)**
```python
# ✅ EXCELLENT (10/10): Comprehensive security
class SecureModelAPI:
    def __init__(self, config: SecurityConfig):
        self._api_key_hash = self._hash_api_key(config.api_key)
        self._rate_limiter = RateLimiter(max_requests=100, window=3600)
        self._input_sanitizer = InputSanitizer()
        self._audit_logger = AuditLogger()
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Secure prediction endpoint with comprehensive protection."""
        
        # Authentication
        if not self._authenticate_request(request.api_key):
            self._audit_logger.log_unauthorized_access(request.source_ip)
            raise UnauthorizedError("Invalid API key")
        
        # Rate limiting
        if not self._rate_limiter.allow_request(request.source_ip):
            raise RateLimitExceededError("Too many requests")
        
        # Input validation and sanitization
        sanitized_input = self._input_sanitizer.sanitize(request.data)
        self._validate_input_schema(sanitized_input)
        
        # Size limits to prevent DoS
        if len(sanitized_input) > MAX_INPUT_SIZE:
            raise InputTooLargeError("Input exceeds maximum size")
        
        try:
            # Secure prediction in isolated environment
            with self._create_secure_context() as context:
                prediction = self._model.predict(sanitized_input, context)
                
            return PredictionResponse(prediction=prediction)
            
        except Exception as e:
            self._audit_logger.log_error(e, request.user_id)
            raise PredictionError("Prediction failed") from e

# ❌ POOR (0/10): Major security issues
def bad_predict(data):
    # No authentication, no validation, direct eval - DANGEROUS!
    result = eval(data)  # Code injection vulnerability!
    return result
```

#### **Data Privacy & Compliance (5 points)**
- [ ] **5/5:** Full GDPR/privacy compliance, data anonymization, audit trails
- [ ] **4/5:** Good privacy practices, minor gaps
- [ ] **3/5:** Basic privacy consideration
- [ ] **1/5:** Minimal privacy protection
- [ ] **0/5:** No privacy consideration

### 2.4 **Testing & Validation Assessment (10 points)**

#### **Test Coverage & Quality (10 points)**
```python
# ✅ EXCELLENT (10/10): Comprehensive testing
class TestModelTrainer:
    """Comprehensive test suite with multiple testing strategies."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Setup mock dependencies for testing."""
        return {
            'data_loader': Mock(spec=DataLoader),
            'preprocessor': Mock(spec=DataPreprocessor),
            'model_factory': Mock(spec=ModelFactory),
            'evaluator': Mock(spec=ModelEvaluator)
        }
    
    def test_successful_training_flow(self, mock_dependencies):
        """Test normal training flow with proper mocking."""
        # Arrange
        trainer = ModelTrainer(**mock_dependencies)
        config = TrainingConfig(model_type="random_forest")
        
        mock_dependencies['data_loader'].load.return_value = create_sample_data()
        mock_dependencies['model_factory'].create.return_value = Mock()
        
        # Act
        result = trainer.train_model(config)
        
        # Assert
        assert result is not None
        assert result.metrics.accuracy > 0
        mock_dependencies['data_loader'].load.assert_called_once()
        mock_dependencies['evaluator'].evaluate.assert_called_once()
    
    def test_training_with_invalid_data(self, mock_dependencies):
        """Test error handling with invalid data."""
        trainer = ModelTrainer(**mock_dependencies)
        config = TrainingConfig(model_type="random_forest")
        
        # Setup error condition
        mock_dependencies['data_loader'].load.side_effect = DataValidationError("Invalid data")
        
        # Test exception handling
        with pytest.raises(TrainingFailedError) as exc_info:
            trainer.train_model(config)
        
        assert "Data validation failed" in str(exc_info.value)
```

## 3. 🚫 **CRITICAL REVIEW BLOCKERS**

### 3.1 **Immediate Rejection Criteria**
- [ ] **Security vulnerabilities** (code injection, hardcoded secrets)
- [ ] **Data leakage** in ML pipelines
- [ ] **No type hints** on public interfaces
- [ ] **Bare except clauses** without specific handling
- [ ] **Hardcoded file paths or URLs**
- [ ] **Missing tests** for new functionality
- [ ] **Breaking changes** without version updates
- [ ] **Memory leaks** or resource leaks
- [ ] **API breaking changes** without deprecation
- [ ] **Compliance violations** (license, privacy)

### 3.2 **Review Completion Blockers**
```python
# ❌ BLOCKERS: These must be fixed before approval

# 1. Security vulnerability
password = "hardcoded_password_123"  # BLOCKER: Hardcoded secret

# 2. Data leakage
scaler.fit(full_dataset)  # BLOCKER: Data leakage
train, test = split(full_dataset)

# 3. No error handling
result = risky_operation()  # BLOCKER: No exception handling

# 4. No type hints
def important_function(data):  # BLOCKER: Missing type hints
    return process(data)

# 5. Resource leak
file = open("data.txt")  # BLOCKER: File not closed
data = file.read()
```

## 4. 📊 **AUTOMATED REVIEW TOOLS INTEGRATION**

### 4.1 **Required Tool Chain**
```yaml
# .github/workflows/code-review.yml
name: Automated Code Review
on: [pull_request]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # Code formatting
      - name: Check Black formatting
        run: black --check src/ tests/
        
      # Type checking
      - name: Type checking with mypy
        run: mypy src/
        
      # Security scanning
      - name: Security scan with bandit
        run: bandit -r src/ -f json -o security-report.json
        
      # Test coverage
      - name: Run tests with coverage
        run: |
          pytest tests/ --cov=src --cov-report=xml --cov-fail-under=90
          
      # AI/ML specific checks
      - name: Check for data leakage
        run: python scripts/check_data_leakage.py
```

## 5. 📋 **REVIEW COMPLETION CHECKLIST**

### **Before Approving ANY Code Review:**
- [ ] All automated checks pass (black, mypy, flake8, tests)
- [ ] Security scan shows no vulnerabilities
- [ ] No data leakage in ML pipelines
- [ ] Type hints on all public interfaces
- [ ] Comprehensive error handling with specific exceptions
- [ ] Test coverage ≥90% for new code
- [ ] Documentation updated for API changes
- [ ] Performance impact assessed and acceptable
- [ ] Memory usage within acceptable limits
- [ ] Breaking changes properly versioned and documented
- [ ] Related tests updated and passing
- [ ] Architecture integrity maintained
- [ ] Dependencies properly managed
- [ ] Logging and monitoring adequate
- [ ] Code follows project conventions consistently

### **AI/ML Specific Checklist:**
- [ ] No data leakage between train/test sets
- [ ] Model interfaces properly abstracted
- [ ] Experiment tracking implemented
- [ ] Model artifacts properly versioned
- [ ] Metrics calculation verified
- [ ] Data validation implemented
- [ ] Pipeline integrity maintained
- [ ] Resource usage optimized for production
- [ ] Model explainability considered
- [ ] Bias and fairness evaluation included

### **Final Approval Criteria:**
- [ ] **Minimum 85/100** overall quality score
- [ ] **Zero security vulnerabilities**
- [ ] **All tests passing**
- [ ] **Performance benchmarks met**
- [ ] **Documentation complete**
- [ ] **Architectural consistency maintained**

## 6. 🎯 **REVIEW SCORING SYSTEM**

### **Quality Score Calculation:**
```
Total Score = Code Quality (30) + AI/ML Specific (25) + Security (15) + Testing (10) + Architecture (20)

Grade Scale:
A+ (95-100): Exceptional quality, production ready
A  (90-94):  High quality, minor improvements needed
B+ (85-89):  Good quality, some improvements needed
B  (80-84):  Adequate quality, several improvements needed
C+ (75-79):  Below standard, significant improvements needed
F  (<75):    Unacceptable, major rework required

Minimum passing score: 85/100 (B+ grade)
```

### **Priority Actions by Score:**
- **95-100:** Approve immediately, use as example for team
- **90-94:** Approve with minor suggestions
- **85-89:** Request improvements, approve after fixes
- **80-84:** Request significant improvements before re-review
- **75-79:** Major rework required, architectural discussion needed
- **<75:** Reject, fundamental redesign required




---
description: 
globs: 
alwaysApply: true
---
---
description: Comprehensive design patterns guide for AI/ML projects with SOLID principles, performance optimization, and industry best practices.
globs: src/**/*.py, tests/**/*.py, architecture/**/*.py
alwaysApply: true
---

# 🎨 DESIGN PATTERNS FOR AI/ML PROJECTS

## 1. 🏗️ **CREATIONAL PATTERNS FOR AI/ML**

### 1.1 **Model Factory Pattern**
```python
# ✅ EXCELLENT: Comprehensive Model Factory for AI/ML
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
import inspect

class ModelInterface(ABC):
    """Abstract base for all ML models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass

class ModelFactory:
    """Factory for creating ML models with proper registration."""
    
    _models: Dict[str, Type[ModelInterface]] = {}
    _default_params: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        default_params: Optional[Dict[str, Any]] = None
    ):
        """Decorator to register model classes."""
        def decorator(model_class: Type[ModelInterface]):
            # Validate model implements interface
            if not issubclass(model_class, ModelInterface):
                raise ValueError(f"Model {model_class} must implement ModelInterface")
            
            cls._models[name] = model_class
            cls._default_params[name] = default_params or {}
            return model_class
        return decorator
    
    @classmethod
    def create_model(
        cls, 
        model_type: str, 
        **kwargs
    ) -> ModelInterface:
        """Create model instance with validation."""
        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model '{model_type}'. Available: {available}")
        
        model_class = cls._models[model_type]
        
        # Merge default params with provided params
        params = cls._default_params[model_type].copy()
        params.update(kwargs)
        
        # Validate parameters match constructor signature
        sig = inspect.signature(model_class.__init__)
        valid_params = {k: v for k, v in params.items() 
                       if k in sig.parameters or 'kwargs' in sig.parameters}
        
        return model_class(**valid_params)

# Usage examples
@ModelFactory.register("random_forest", {"n_estimators": 100, "random_state": 42})
class RandomForestModel(ModelInterface):
    """Random Forest classifier with optimal defaults."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before prediction")
        return self._model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state
        }
```

### 1.2 **Pipeline Builder Pattern**
```python
# ✅ EXCELLENT: AI/ML Pipeline Builder
from typing import List, Tuple, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class PipelineStep:
    """Represents a step in the ML pipeline."""
    name: str
    transformer: Any
    fit_params: Dict[str, Any] = None
    transform_params: Dict[str, Any] = None

class MLPipelineBuilder:
    """Builder for creating complex ML pipelines."""
    
    def __init__(self):
        self._steps: List[PipelineStep] = []
        self._validation_steps: List[Callable] = []
        self._metadata: Dict[str, Any] = {}
    
    def add_preprocessing_step(
        self, 
        name: str, 
        transformer: Any,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> 'MLPipelineBuilder':
        """Add data preprocessing step."""
        step = PipelineStep(
            name=f"preprocess_{name}",
            transformer=transformer,
            fit_params=fit_params or {}
        )
        self._steps.append(step)
        return self
    
    def add_feature_engineering_step(
        self,
        name: str,
        transformer: Any,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> 'MLPipelineBuilder':
        """Add feature engineering step."""
        step = PipelineStep(
            name=f"feature_{name}",
            transformer=transformer,
            fit_params=fit_params or {}
        )
        self._steps.append(step)
        return self
    
    def add_model_step(
        self,
        model: ModelInterface,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> 'MLPipelineBuilder':
        """Add model training step."""
        step = PipelineStep(
            name="model",
            transformer=model,
            fit_params=fit_params or {}
        )
        self._steps.append(step)
        return self
    
    def add_validation(
        self,
        validation_func: Callable[[Any], bool]
    ) -> 'MLPipelineBuilder':
        """Add validation step."""
        self._validation_steps.append(validation_func)
        return self
    
    def with_metadata(
        self,
        **metadata
    ) -> 'MLPipelineBuilder':
        """Add metadata to pipeline."""
        self._metadata.update(metadata)
        return self
    
    def build(self) -> 'MLPipeline':
        """Build the final pipeline."""
        if not self._steps:
            raise ValueError("Pipeline must have at least one step")
        
        # Validate pipeline structure
        model_steps = [s for s in self._steps if s.name == "model"]
        if len(model_steps) != 1:
            raise ValueError("Pipeline must have exactly one model step")
        
        return MLPipeline(
            steps=self._steps.copy(),
            validations=self._validation_steps.copy(),
            metadata=self._metadata.copy()
        )
    
    def reset(self) -> 'MLPipelineBuilder':
        """Reset builder for reuse."""
        self._steps.clear()
        self._validation_steps.clear()
        self._metadata.clear()
        return self

class MLPipeline:
    """Complete ML pipeline with validation and metadata."""
    
    def __init__(
        self,
        steps: List[PipelineStep],
        validations: List[Callable],
        metadata: Dict[str, Any]
    ):
        self.steps = steps
        self.validations = validations
        self.metadata = metadata
        self._fitted_steps: Dict[str, Any] = {}
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPipeline':
        """Fit the entire pipeline."""
        current_X, current_y = X, y
        
        for step in self.steps:
            try:
                if hasattr(step.transformer, 'fit'):
                    if step.name == "model":
                        fitted_transformer = step.transformer.fit(current_X, current_y)
                    else:
                        fitted_transformer = step.transformer.fit(current_X)
                        if hasattr(fitted_transformer, 'transform'):
                            current_X = fitted_transformer.transform(current_X)
                else:
                    fitted_transformer = step.transformer
                
                self._fitted_steps[step.name] = fitted_transformer
                
                # Run validations
                for validation in self.validations:
                    if not validation(fitted_transformer):
                        raise ValidationError(f"Validation failed for step {step.name}")
                        
            except Exception as e:
                raise PipelineError(f"Failed to fit step '{step.name}': {e}") from e
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using fitted pipeline."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        current_X = X
        
        # Apply all preprocessing and feature engineering steps
        for step in self.steps[:-1]:  # All except model
            transformer = self._fitted_steps[step.name]
            if hasattr(transformer, 'transform'):
                current_X = transformer.transform(current_X)
        
        # Apply model prediction
        model = self._fitted_steps["model"]
        return model.predict(current_X)
```

### 1.3 **Configuration Factory Pattern**
```python
# ✅ EXCELLENT: Configuration Factory for different environments
from pydantic import BaseModel, Field
from typing import Union, Literal
from pathlib import Path

class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    learning_rate: float = Field(gt=0, le=1)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0, le=10000)
    early_stopping_patience: int = Field(gt=0)
    model_type: str
    random_seed: int = Field(ge=0)

class DataConfig(BaseModel):
    """Data configuration with validation."""
    data_path: Path
    target_column: str
    test_size: float = Field(gt=0, lt=1)
    validation_size: float = Field(gt=0, lt=1)
    stratify: bool = True

class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    training: TrainingConfig
    data: DataConfig
    experiment_name: str
    output_dir: Path
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

class ConfigurationFactory:
    """Factory for creating different configuration types."""
    
    @staticmethod
    def create_development_config() -> ExperimentConfig:
        """Create configuration optimized for development."""
        return ExperimentConfig(
            training=TrainingConfig(
                learning_rate=0.01,
                batch_size=32,
                epochs=10,  # Small for quick iterations
                early_stopping_patience=3,
                model_type="random_forest",
                random_seed=42
            ),
            data=DataConfig(
                data_path=Path("data/dev_sample.csv"),
                target_column="target",
                test_size=0.2,
                validation_size=0.2
            ),
            experiment_name="development_experiment",
            output_dir=Path("outputs/dev"),
            logging_level="DEBUG"
        )
    
    @staticmethod
    def create_production_config() -> ExperimentConfig:
        """Create configuration optimized for production."""
        return ExperimentConfig(
            training=TrainingConfig(
                learning_rate=0.001,
                batch_size=128,
                epochs=1000,  # More epochs for better performance
                early_stopping_patience=50,
                model_type="neural_network",
                random_seed=42
            ),
            data=DataConfig(
                data_path=Path("data/full_dataset.csv"),
                target_column="target",
                test_size=0.15,
                validation_size=0.15
            ),
            experiment_name="production_model",
            output_dir=Path("outputs/prod"),
            logging_level="INFO"
        )
    
    @staticmethod
    def create_hyperparameter_tuning_config() -> ExperimentConfig:
        """Create configuration for hyperparameter tuning."""
        return ExperimentConfig(
            training=TrainingConfig(
                learning_rate=0.01,  # Will be overridden by tuning
                batch_size=64,
                epochs=100,
                early_stopping_patience=10,
                model_type="gradient_boosting",
                random_seed=42
            ),
            data=DataConfig(
                data_path=Path("data/tuning_dataset.csv"),
                target_column="target",
                test_size=0.2,
                validation_size=0.2
            ),
            experiment_name="hyperparameter_tuning",
            output_dir=Path("outputs/tuning"),
            logging_level="INFO"
        )
```

## 2. 🔗 **STRUCTURAL PATTERNS FOR AI/ML**

### 2.1 **Adapter Pattern for Data Sources**
```python
# ✅ EXCELLENT: Data Source Adapter Pattern
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DataSourceInterface(ABC):
    """Common interface for all data sources."""
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data and return as DataFrame."""
        pass
    
    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate data schema."""
        pass

class CSVDataSource(DataSourceInterface):
    """CSV data source implementation."""
    
    def __init__(self, file_path: Path, **kwargs):
        self.file_path = file_path
        self.read_kwargs = kwargs
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data."""
        return pd.read_csv(self.file_path, **self.read_kwargs)
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate CSV schema."""
        # Implement CSV-specific validation
        return not df.empty and len(df.columns) > 0

class DatabaseAdapter(DataSourceInterface):
    """Adapter for database data sources."""
    
    def __init__(self, connection_string: str, query: str):
        self.connection_string = connection_string
        self.query = query
    
    def load_data(self) -> pd.DataFrame:
        """Load data from database."""
        import sqlalchemy
        engine = sqlalchemy.create_engine(self.connection_string)
        return pd.read_sql(self.query, engine)
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate database schema."""
        # Implement database-specific validation
        return not df.empty and len(df.columns) > 0

class APIAdapter(DataSourceInterface):
    """Adapter for API data sources."""
    
    def __init__(self, api_url: str, headers: Optional[Dict[str, str]] = None):
        self.api_url = api_url
        self.headers = headers or {}
    
    def load_data(self) -> pd.DataFrame:
        """Load data from API."""
        import requests
        response = requests.get(self.api_url, headers=self.headers)
        response.raise_for_status()
        
        # Convert JSON response to DataFrame
        return pd.DataFrame(response.json())
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate API response schema."""
        # Implement API-specific validation
        return not df.empty and len(df.columns) > 0

class UniversalDataLoader:
    """Context class that uses adapters to load data from any source."""
    
    def __init__(self, data_source: DataSourceInterface):
        self.data_source = data_source
    
    def load_and_validate(self) -> pd.DataFrame:
        """Load data using adapter and validate."""
        df = self.data_source.load_data()
        
        if not self.data_source.validate_schema(df):
            raise DataValidationError("Data schema validation failed")
        
        return df

# Usage
csv_loader = UniversalDataLoader(CSVDataSource(Path("data.csv")))
db_loader = UniversalDataLoader(DatabaseAdapter("sqlite:///db.sqlite", "SELECT * FROM table"))
api_loader = UniversalDataLoader(APIAdapter("https://api.example.com/data"))
```

### 2.2 **Decorator Pattern for Model Enhancement**
```python
# ✅ EXCELLENT: Model Enhancement Decorators
from functools import wraps
import time
from typing import Any, Callable

class ModelDecorator(ModelInterface):
    """Base decorator for model enhancements."""
    
    def __init__(self, model: ModelInterface):
        self._model = model
        self._metadata: Dict[str, Any] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelDecorator':
        """Delegate fitting to wrapped model."""
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Delegate prediction to wrapped model."""
        return self._model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters from wrapped model."""
        return self._model.get_params()

class PerformanceMonitoringDecorator(ModelDecorator):
    """Decorator that adds performance monitoring."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PerformanceMonitoringDecorator':
        """Fit model with timing."""
        start_time = time.time()
        result = super().fit(X, y)
        end_time = time.time()
        
        self._metadata["fit_time"] = end_time - start_time
        self._metadata["training_samples"] = len(X)
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with timing and memory monitoring."""
        import psutil
        import os
        
        # Memory before prediction
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        start_time = time.time()
        predictions = super().predict(X)
        end_time = time.time()
        
        # Memory after prediction
        memory_after = process.memory_info().rss
        
        self._metadata.update({
            "last_prediction_time": end_time - start_time,
            "last_prediction_samples": len(X),
            "memory_increase": memory_after - memory_before
        })
        
        return predictions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._metadata.copy()

class CachingDecorator(ModelDecorator):
    """Decorator that adds prediction caching."""
    
    def __init__(self, model: ModelInterface, cache_size: int = 1000):
        super().__init__(model)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with caching."""
        # Create cache key from input data
        cache_key = hash(X.tobytes())
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Get prediction from wrapped model
        predictions = super().predict(X)
        
        # Cache result if cache not full
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = predictions.copy()
        
        return predictions
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._cache.clear()

class ValidationDecorator(ModelDecorator):
    """Decorator that adds input validation."""
    
    def __init__(self, model: ModelInterface, expected_features: int):
        super().__init__(model)
        self.expected_features = expected_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ValidationDecorator':
        """Fit with input validation."""
        self._validate_input(X, y)
        return super().fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with input validation."""
        self._validate_prediction_input(X)
        return super().predict(X)
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training input."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        if X.shape[1] != self.expected_features:
            raise ValueError(f"Expected {self.expected_features} features, got {X.shape[1]}")
    
    def _validate_prediction_input(self, X: np.ndarray) -> None:
        """Validate prediction input."""
        if X.shape[1] != self.expected_features:
            raise ValueError(f"Expected {self.expected_features} features, got {X.shape[1]}")

# Usage: Stacking decorators
base_model = RandomForestModel(n_estimators=100)
enhanced_model = PerformanceMonitoringDecorator(
    CachingDecorator(
        ValidationDecorator(base_model, expected_features=10)
    )
)
```

## 3. 🎭 **BEHAVIORAL PATTERNS FOR AI/ML**

### 3.1 **Strategy Pattern for Algorithms**
```python
# ✅ EXCELLENT: Algorithm Strategy Pattern
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies."""
    
    def optimize(self, model: ModelInterface, X: np.ndarray, y: np.ndarray) -> ModelInterface:
        """Optimize model parameters."""
        ...

class GridSearchStrategy:
    """Grid search optimization strategy."""
    
    def __init__(self, param_grid: Dict[str, List[Any]], cv: int = 5):
        self.param_grid = param_grid
        self.cv = cv
    
    def optimize(self, model: ModelInterface, X: np.ndarray, y: np.ndarray) -> ModelInterface:
        """Optimize using grid search."""
        from sklearn.model_selection import GridSearchCV
        
        # Wrap model for sklearn compatibility
        sklearn_model = self._wrap_model_for_sklearn(model)
        
        grid_search = GridSearchCV(
            sklearn_model, 
            self.param_grid, 
            cv=self.cv,
            scoring='accuracy'
        )
        
        grid_search.fit(X, y)
        
        # Create optimized model with best parameters
        best_params = grid_search.best_params_
        optimized_model = ModelFactory.create_model(
            model.get_params()["model_type"],
            **best_params
        )
        
        return optimized_model.fit(X, y)

class BayesianOptimizationStrategy:
    """Bayesian optimization strategy."""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
    
    def optimize(self, model: ModelInterface, X: np.ndarray, y: np.ndarray) -> ModelInterface:
        """Optimize using Bayesian optimization."""
        import optuna
        
        def objective(trial):
            # Define hyperparameter search space
            params = self._define_search_space(trial, model)
            
            # Create model with trial parameters
            trial_model = ModelFactory.create_model(
                model.get_params()["model_type"],
                **params
            )
            
            # Evaluate with cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(trial_model, X, y, cv=5)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Create optimized model
        best_params = study.best_params
        optimized_model = ModelFactory.create_model(
            model.get_params()["model_type"],
            **best_params
        )
        
        return optimized_model.fit(X, y)

class ModelOptimizer:
    """Context class for model optimization."""
    
    def __init__(self, strategy: OptimizationStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """Change optimization strategy."""
        self.strategy = strategy
    
    def optimize_model(self, model: ModelInterface, X: np.ndarray, y: np.ndarray) -> ModelInterface:
        """Optimize model using current strategy."""
        return self.strategy.optimize(model, X, y)

# Usage
base_model = RandomForestModel()
optimizer = ModelOptimizer(GridSearchStrategy({"n_estimators": [50, 100, 200]}))

# Optimize with grid search
optimized_model = optimizer.optimize_model(base_model, X_train, y_train)

# Switch to Bayesian optimization
optimizer.set_strategy(BayesianOptimizationStrategy(n_trials=50))
further_optimized = optimizer.optimize_model(optimized_model, X_train, y_train)
```

### 3.2 **Observer Pattern for Training Monitoring**
```python
# ✅ EXCELLENT: Training Progress Observer Pattern
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
from pathlib import Path

class TrainingObserver(ABC):
    """Abstract observer for training events."""
    
    @abstractmethod
    def on_training_start(self, model_info: Dict[str, Any]) -> None:
        """Called when training starts."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_training_end(self, final_metrics: Dict[str, float]) -> None:
        """Called when training ends."""
        pass

class ConsoleLoggingObserver(TrainingObserver):
    """Observer that logs training progress to console."""
    
    def on_training_start(self, model_info: Dict[str, Any]) -> None:
        print(f"🚀 Training started: {model_info['model_type']}")
        print(f"📊 Dataset: {model_info['samples']} samples, {model_info['features']} features")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch:3d} | {metrics_str}")
    
    def on_training_end(self, final_metrics: Dict[str, float]) -> None:
        print("✅ Training completed!")
        print("📈 Final metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

class FileLoggingObserver(TrainingObserver):
    """Observer that logs training progress to file."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.training_log: List[Dict[str, Any]] = []
    
    def on_training_start(self, model_info: Dict[str, Any]) -> None:
        self.training_log = [{
            "event": "training_start",
            "timestamp": time.time(),
            "model_info": model_info
        }]
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.training_log.append({
            "event": "epoch_end",
            "timestamp": time.time(),
            "epoch": epoch,
            "metrics": metrics
        })
        
        # Write to file after each epoch
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
    
    def on_training_end(self, final_metrics: Dict[str, float]) -> None:
        self.training_log.append({
            "event": "training_end",
            "timestamp": time.time(),
            "final_metrics": final_metrics
        })
        
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)

class EarlyStoppingObserver(TrainingObserver):
    """Observer that implements early stopping."""
    
    def __init__(self, patience: int = 10, metric: str = "val_loss", mode: str = "min"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.should_stop = False
    
    def on_training_start(self, model_info: Dict[str, Any]) -> None:
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.wait = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        current_value = metrics.get(self.metric)
        if current_value is None:
            return
        
        improved = (
            (self.mode == 'min' and current_value < self.best_value) or
            (self.mode == 'max' and current_value > self.best_value)
        )
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                print(f"🛑 Early stopping triggered at epoch {epoch}")
    
    def on_training_end(self, final_metrics: Dict[str, float]) -> None:
        pass

class ObservableTrainer:
    """Trainer that supports observers."""
    
    def __init__(self):
        self._observers: List[TrainingObserver] = []
    
    def attach_observer(self, observer: TrainingObserver) -> None:
        """Attach an observer."""
        self._observers.append(observer)
    
    def detach_observer(self, observer: TrainingObserver) -> None:
        """Detach an observer."""
        self._observers.remove(observer)
    
    def _notify_training_start(self, model_info: Dict[str, Any]) -> None:
        """Notify all observers of training start."""
        for observer in self._observers:
            observer.on_training_start(model_info)
    
    def _notify_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Notify all observers of epoch end."""
        for observer in self._observers:
            observer.on_epoch_end(epoch, metrics)
    
    def _notify_training_end(self, final_metrics: Dict[str, float]) -> None:
        """Notify all observers of training end."""
        for observer in self._observers:
            observer.on_training_end(final_metrics)
    
    def train(self, model: ModelInterface, X: np.ndarray, y: np.ndarray) -> ModelInterface:
        """Train model with observer notifications."""
        model_info = {
            "model_type": type(model).__name__,
            "samples": len(X),
            "features": X.shape[1] if len(X.shape) > 1 else 1
        }
        
        self._notify_training_start(model_info)
        
        # Simulate training with epochs (this would be actual training logic)
        for epoch in range(100):
            # Simulate training metrics
            metrics = {
                "loss": 0.5 - epoch * 0.01 + np.random.normal(0, 0.05),
                "accuracy": 0.5 + epoch * 0.005 + np.random.normal(0, 0.02)
            }
            
            self._notify_epoch_end(epoch, metrics)
            
            # Check for early stopping
            early_stopping_observers = [obs for obs in self._observers 
                                      if isinstance(obs, EarlyStoppingObserver)]
            if any(obs.should_stop for obs in early_stopping_observers):
                break
        
        # Fit the actual model
        trained_model = model.fit(X, y)
        
        final_metrics = {"final_accuracy": 0.95, "final_loss": 0.05}
        self._notify_training_end(final_metrics)
        
        return trained_model

# Usage
trainer = ObservableTrainer()
trainer.attach_observer(ConsoleLoggingObserver())
trainer.attach_observer(FileLoggingObserver(Path("training_log.json")))
trainer.attach_observer(EarlyStoppingObserver(patience=5, metric="loss"))

model = trainer.train(RandomForestModel(), X_train, y_train)
```

## 4. 🚫 **AI/ML ANTI-PATTERNS TO AVOID**

### 4.1 **Common Anti-Patterns**
```python
# ❌ ANTI-PATTERN: God Class Pattern
class MLGodClass:
    """Violates SRP - does everything poorly."""
    def load_data(self): pass
    def clean_data(self): pass
    def engineer_features(self): pass
    def train_model(self): pass
    def evaluate_model(self): pass
    def deploy_model(self): pass
    def monitor_model(self): pass
    # ... 50 more methods

# ✅ CORRECT: Separate responsibilities
class DataLoader: pass
class DataCleaner: pass
class FeatureEngineer: pass
class ModelTrainer: pass

# ❌ ANTI-PATTERN: Hardcoded Factory
class BadModelFactory:
    def create_model(self, model_type):
        if model_type == "rf":
            return RandomForest(n_estimators=100)  # Hardcoded params
        elif model_type == "svm":
            return SVM(C=1.0)  # No flexibility
        # No extensibility for new models

# ✅ CORRECT: Use registry pattern shown above

# ❌ ANTI-PATTERN: Data Leakage in Pipeline
class LeakyPipeline:
    def fit(self, X, y):
        # WRONG: Fit on entire dataset before split
        self.scaler.fit(X)
        X_train, X_test = train_test_split(X)
        return self

# ✅ CORRECT: Proper pipeline shown above
```

### 4.2 **Performance Anti-Patterns**
```python
# ❌ ANTI-PATTERN: Inefficient Data Loading
def bad_data_loading():
    data = []
    for file in files:
        df = pd.read_csv(file)
        data.append(df)  # Memory inefficient
    return pd.concat(data)

# ✅ CORRECT: Streaming approach
def good_data_loading():
    def data_generator():
        for file in files:
            yield pd.read_csv(file)
    return data_generator()

# ❌ ANTI-PATTERN: No caching of expensive operations
def bad_feature_engineering(data):
    # Recalculates every time
    complex_features = expensive_calculation(data)
    return complex_features

# ✅ CORRECT: Use caching decorator shown above
```

## 5. 📋 **PATTERN IMPLEMENTATION CHECKLIST**

### **Before implementing any pattern:**
- [ ] **Clear problem definition** - What specific problem does this pattern solve?
- [ ] **Alternative evaluation** - Why is this pattern better than alternatives?
- [ ] **SOLID compliance** - Does implementation follow SOLID principles?
- [ ] **Type safety** - Are all interfaces properly typed?
- [ ] **Error handling** - Are edge cases and errors handled?
- [ ] **Testing strategy** - How will pattern be tested?
- [ ] **Documentation** - Is pattern usage documented with examples?
- [ ] **Performance impact** - What are the performance implications?
- [ ] **Extensibility** - Can pattern be extended for future needs?
- [ ] **Integration** - How does pattern integrate with existing code?

### **Pattern Quality Gates:**
- [ ] **90%+ test coverage** for pattern implementation
- [ ] **Zero security vulnerabilities** in pattern code
- [ ] **Performance benchmarks** established and met
- [ ] **Memory usage** within acceptable limits
- [ ] **Thread safety** considered and implemented if needed
- [ ] **Configuration flexibility** without hardcoded values
- [ ] **Logging and monitoring** integrated appropriately
- [ ] **Error messages** are clear and actionable
- [ ] **API compatibility** maintained during pattern evolution
- [ ] **Documentation completeness** verified

### **AI/ML Specific Validations:**
- [ ] **No data leakage** in any pipeline patterns
- [ ] **Model interface consistency** across all implementations
- [ ] **Experiment reproducibility** enabled by patterns
- [ ] **Resource cleanup** properly implemented
- [ ] **Scalability considerations** for large datasets
- [ ] **Model versioning** support in relevant patterns
- [ ] **Bias and fairness** considerations documented
- [ ] **Deployment readiness** of pattern implementations

## 6. 🎯 **PATTERN SELECTION GUIDE**

### **Use Factory Pattern when:**
- Creating multiple types of models/algorithms
- Need dynamic model selection based on configuration
- Want to encapsulate object creation logic
- Multiple model types share common interface

### **Use Builder Pattern when:**
- Creating complex ML pipelines with multiple steps
- Need fluent interface for configuration
- Object construction has many optional parameters
- Want to ensure object is properly configured before use

### **Use Strategy Pattern when:**
- Multiple algorithms solve same problem differently
- Want to switch algorithms at runtime
- Need to make algorithms interchangeable
- Want to isolate algorithm-specific code

### **Use Observer Pattern when:**
- Multiple components need to react to training events
- Want to decouple training logic from monitoring/logging
- Need flexible notification system
- Want to add new observers without modifying core code

### **Use Decorator Pattern when:**
- Want to add functionality to models without inheritance
- Need to compose multiple enhancements
- Want to add cross-cutting concerns (logging, caching)
- Need to maintain original interface

### **Use Adapter Pattern when:**
- Integrating different data sources or APIs
- Need to make incompatible interfaces work together
- Want to isolate external dependencies
- Legacy system integration required



---
description: 
globs: 
alwaysApply: true
---
# 📋 GENERAL CODING GUIDELINES FOR AI/ML PROJECTS

## 1. 🎯 **FUNDAMENTAL DEVELOPMENT PRINCIPLES**

### 1.1 **SOLID Principles Application**
```python
# ✅ Single Responsibility Principle (SRP)
class DataValidator:
    """Validates data quality and schema compliance."""
    def validate_schema(self, data: pd.DataFrame) -> bool: pass
    def check_data_quality(self, data: pd.DataFrame) -> QualityReport: pass

class ModelTrainer:
    """Handles model training operations."""
    def train(self, model: Model, data: TrainingData) -> TrainedModel: pass

# ❌ Violates SRP - Multiple responsibilities
class DataProcessor:
    def validate_data(self): pass
    def train_model(self): pass
    def save_results(self): pass
    def send_notifications(self): pass  # Too many responsibilities!

# ✅ Open/Closed Principle (OCP)
from abc import ABC, abstractmethod

class MetricCalculator(ABC):
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: pass

class AccuracyCalculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

# ✅ Dependency Inversion Principle (DIP)
class ModelEvaluator:
    def __init__(self, metrics: List[MetricCalculator]):
        self.metrics = metrics  # Depends on abstraction, not concretions
```

### 1.2 **Clean Code Principles**
```python
# ✅ DRY (Don't Repeat Yourself)
class ConfigurationManager:
    @staticmethod
    def get_db_config() -> Dict[str, str]:
        """Single source of truth for database configuration."""
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "ml_project")
        }

# ❌ WET (Write Everything Twice)
def connect_to_training_db():
    host = os.getenv("DB_HOST", "localhost")  # Repeated code
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "ml_project")

def connect_to_inference_db():
    host = os.getenv("DB_HOST", "localhost")  # Same code repeated
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "ml_project")

# ✅ KISS (Keep It Simple, Stupid)
def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Simple, clear accuracy calculation."""
    return np.mean(predictions == labels)

# ❌ Over-complicated
def calculate_accuracy_complex(predictions, labels):
    correct = 0
    total = len(predictions)
    for i in range(total):
        if predictions[i] == labels[i]:
            correct += 1
    return correct / total if total > 0 else 0.0

# ✅ YAGNI (You Aren't Gonna Need It)
class SimpleModelTrainer:
    """Implements only what we need now."""
    def train(self, model, data): pass
    def evaluate(self, model, data): pass

# ❌ Over-engineering for future needs
class OverEngineeredTrainer:
    def train(self): pass
    def train_distributed(self): pass      # Not needed yet
    def train_with_gpu_cluster(self): pass # Not needed yet
    def train_with_quantum_ai(self): pass  # Definitely not needed!
```

## 2. 📝 **CODE ORGANIZATION & STRUCTURE**

### 2.1 **Project Structure Standards**
```
project_root/
├── src/                          # Source code
│   ├── data/                     # Data handling modules
│   │   ├── loaders/             # Data loading utilities
│   │   ├── processors/          # Data processing pipelines
│   │   └── validators/          # Data validation logic
│   ├── models/                   # Model definitions
│   │   ├── architectures/       # Model architectures
│   │   ├── training/           # Training logic
│   │   └── evaluation/         # Evaluation utilities
│   ├── features/                # Feature engineering
│   ├── utils/                   # Utility functions
│   ├── config/                  # Configuration management
│   └── api/                     # API endpoints
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
├── data/                       # Data directory
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── external/               # External data
├── models/                     # Trained models
├── outputs/                    # Output artifacts
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker configuration
└── README.md                   # Project documentation
```

### 2.2 **Import Organization**
```python
# ✅ CORRECT: Import order (PEP 8)
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union

# 2. Related third-party imports
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

# 3. Local application/library imports
from src.data.loaders import DataLoader
from src.models.base import BaseModel
from src.utils.logging import get_logger

# ❌ WRONG: Mixed import order
import pandas as pd
from src.models.base import BaseModel
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
```

## 3. 🔐 **SECURITY & CONFIGURATION**

### 3.1 **Environment Variables & Secrets**
```python
# ✅ CORRECT: Secure configuration management
import os
from typing import Optional

class SecurityConfig:
    """Secure configuration with validation."""
    
    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> str:
        """Get secret from environment with validation."""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    @staticmethod
    def get_database_url() -> str:
        """Get database URL from environment."""
        return SecurityConfig.get_secret("DATABASE_URL")
    
    @staticmethod
    def get_api_key() -> str:
        """Get API key from environment."""
        return SecurityConfig.get_secret("OPENAI_API_KEY")

# ❌ WRONG: Hardcoded secrets
class BadConfig:
    DATABASE_URL = "postgresql://user:password123@localhost/db"  # Security risk!
    API_KEY = "sk-abc123def456"  # Never hardcode keys!

# ✅ CORRECT: .env file usage
"""
# .env file (never commit to version control)
DATABASE_URL=postgresql://user:secure_password@localhost/db
OPENAI_API_KEY=sk-your-secret-key-here
ENVIRONMENT=development
LOG_LEVEL=DEBUG
"""
```

### 3.2 **Input Validation & Sanitization**
```python
# ✅ CORRECT: Comprehensive input validation
from pydantic import BaseModel, validator, Field
from typing import List, Optional

class ModelInput(BaseModel):
    """Validated model input with security checks."""
    
    features: List[float] = Field(..., min_items=1, max_items=1000)
    model_name: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$', max_length=50)
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature values are reasonable."""
        if any(abs(x) > 1e6 for x in v):  # Prevent extreme values
            raise ValueError("Feature values too large")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name is safe."""
        if '..' in v or '/' in v:  # Prevent path traversal
            raise ValueError("Invalid model name")
        return v

# ❌ WRONG: No input validation
def bad_predict(features, model_name):
    # Direct use without validation - security risk!
    model = load_model(f"/models/{model_name}")  # Path traversal vulnerability
    return model.predict(features)  # No input validation
```

## 4. 📊 **LOGGING & MONITORING**

### 4.1 **Structured Logging**
```python
# ✅ CORRECT: Structured logging for AI/ML
import logging
import structlog
from typing import Any, Dict

def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for AI/ML projects."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

class MLLogger:
    """Structured logger for ML operations."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def log_training_start(self, model_info: Dict[str, Any]) -> None:
        """Log training start with context."""
        self.logger.info(
            "Training started",
            model_type=model_info.get("type"),
            dataset_size=model_info.get("dataset_size"),
            features=model_info.get("num_features"),
            hyperparameters=model_info.get("hyperparameters")
        )
    
    def log_prediction(self, input_shape: tuple, prediction_time: float) -> None:
        """Log prediction with performance metrics."""
        self.logger.info(
            "Prediction completed",
            input_shape=input_shape,
            prediction_time_ms=prediction_time * 1000,
            throughput_per_second=input_shape[0] / prediction_time if prediction_time > 0 else 0
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log errors with full context."""
        self.logger.error(
            "Operation failed",
            error_type=type(error).__name__,
            error_message=str(error),
            **context
        )

# ❌ WRONG: Basic print statements
def bad_logging():
    print("Training started")  # No context, no structure
    print(f"Error: {e}")       # No proper error handling
```

### 4.2 **Performance Monitoring**
```python
# ✅ CORRECT: Performance monitoring decorators
import time
import psutil
import functools
from typing import Callable, Any

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Timing
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Memory after
        memory_after = process.memory_info().rss
        
        # Log performance metrics
        logger = structlog.get_logger(func.__module__)
        logger.info(
            "Function performance",
            function_name=func.__name__,
            execution_time_seconds=end_time - start_time,
            memory_increase_mb=(memory_after - memory_before) / 1024 / 1024,
            args_count=len(args),
            kwargs_count=len(kwargs)
        )
        
        return result
    
    return wrapper

@monitor_performance
def train_model(X: np.ndarray, y: np.ndarray) -> Model:
    """Train model with automatic performance monitoring."""
    # Training logic here
    pass
```

## 5. 🧪 **TESTING FUNDAMENTALS**

### 5.1 **Test Organization**
```python
# ✅ CORRECT: Comprehensive test structure
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.models.classifier import RandomForestClassifier

class TestRandomForestClassifier:
    """Comprehensive test suite for RandomForest classifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide consistent test data."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Provide pre-trained model for tests."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        return model.fit(X, y)
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = RandomForestClassifier()
        assert model.n_estimators == 100
        assert model.random_state is None
    
    def test_model_training(self, sample_data):
        """Test model training with valid data."""
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        
        trained_model = model.fit(X, y)
        
        assert trained_model.is_fitted
        assert hasattr(trained_model, '_sklearn_model')
    
    def test_model_prediction(self, trained_model, sample_data):
        """Test model prediction functionality."""
        X, _ = sample_data
        predictions = trained_model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_invalid_input_handling(self):
        """Test model handles invalid input gracefully."""
        model = RandomForestClassifier()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(np.random.rand(10, 5))
    
    @pytest.mark.parametrize("n_estimators,expected_type", [
        (50, int),
        (100, int),
        (200, int)
    ])
    def test_parametrized_training(self, sample_data, n_estimators, expected_type):
        """Test training with different parameters."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X, y)
        
        assert isinstance(model.n_estimators, expected_type)
        assert model.n_estimators == n_estimators

# ❌ WRONG: Poor testing practices
def test_something():
    assert True  # Meaningless test

def test_model():
    model = RandomForestClassifier()
    # No proper setup, no assertions, no error handling
```

## 6. 📈 **PERFORMANCE & OPTIMIZATION**

### 6.1 **Memory Management**
```python
# ✅ CORRECT: Memory-efficient data processing
from typing import Generator, Iterator
import gc

def process_large_dataset(
    data_source: str, 
    batch_size: int = 1000
) -> Generator[pd.DataFrame, None, None]:
    """Process large datasets in memory-efficient batches."""
    
    chunk_iterator = pd.read_csv(data_source, chunksize=batch_size)
    
    for chunk in chunk_iterator:
        # Process chunk
        processed_chunk = preprocess_data(chunk)
        
        yield processed_chunk
        
        # Explicit memory cleanup
        del chunk
        del processed_chunk
        gc.collect()

def efficient_model_inference(
    model: Model, 
    data: np.ndarray, 
    batch_size: int = 100
) -> np.ndarray:
    """Efficient batch inference for large datasets."""
    
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_predictions = model.predict(batch)
        results.append(batch_predictions)
        
        # Clear batch from memory
        del batch
    
    return np.concatenate(results)

# ❌ WRONG: Memory inefficient
def bad_data_processing():
    # Loading entire dataset into memory
    all_data = pd.read_csv("huge_dataset.csv")  # Memory explosion!
    processed_data = []
    
    for row in all_data.iterrows():
        processed_data.append(process_row(row))  # Growing list
    
    return processed_data  # Doubles memory usage
```

### 6.2 **Algorithmic Efficiency**
```python
# ✅ CORRECT: Efficient algorithms and data structures
from collections import defaultdict, Counter
import bisect

class EfficientDataProcessor:
    """Demonstrates efficient algorithm choices."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._sorted_keys: List[str] = []
    
    def find_similar_items(self, item: str, items: List[str]) -> List[str]:
        """Efficient similarity search using sets."""
        item_set = set(item.lower().split())
        similarities = []
        
        for candidate in items:
            candidate_set = set(candidate.lower().split())
            similarity = len(item_set & candidate_set) / len(item_set | candidate_set)
            
            if similarity > 0.5:
                similarities.append((candidate, similarity))
        
        # Sort by similarity (most similar first)
        return [item for item, _ in sorted(similarities, key=lambda x: x[1], reverse=True)]
    
    def efficient_counting(self, items: List[str]) -> Dict[str, int]:
        """Use Counter for efficient counting."""
        return Counter(items)  # O(n) vs manual loop O(n) but optimized
    
    def efficient_grouping(self, items: List[tuple]) -> Dict[str, List]:
        """Use defaultdict for efficient grouping."""
        groups = defaultdict(list)
        for key, value in items:
            groups[key].append(value)
        return dict(groups)

# ❌ WRONG: Inefficient algorithms
class InefficiientProcessor:
    def find_similar_items(self, item: str, items: List[str]) -> List[str]:
        """Inefficient O(n²) similarity calculation."""
        similarities = []
        for candidate in items:
            similarity = 0
            for word in item.split():
                if word in candidate:  # Inefficient string search
                    similarity += 1
            similarities.append((candidate, similarity))
        
        # Inefficient sorting
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][1] < similarities[j][1]:
                    similarities[i], similarities[j] = similarities[j], similarities[i]
        
        return [item for item, _ in similarities]
```

## 7. 🔄 **VERSION CONTROL & COLLABORATION**

### 7.1 **Git Best Practices**
```bash
# ✅ CORRECT: Semantic commit messages
git commit -m "feat: add Random Forest model implementation

- Implement RandomForestClassifier with sklearn backend
- Add comprehensive parameter validation
- Include performance monitoring decorators
- Add unit tests with 95% coverage

Closes #123"

git commit -m "fix: resolve data leakage in preprocessing pipeline

- Move scaler fitting after train/test split
- Update pipeline tests to catch data leakage
- Add validation checks for proper data isolation

Fixes #456"

git commit -m "docs: update API documentation for model factory

- Add type hints to all factory methods
- Include usage examples in docstrings
- Update README with factory pattern explanation"

# ❌ WRONG: Poor commit messages
git commit -m "fix stuff"
git commit -m "updates"
git commit -m "wip"
```

### 7.2 **Branch Strategy**
```bash
# ✅ CORRECT: Git Flow branch strategy
main                    # Production-ready code
├── develop            # Integration branch
├── feature/model-factory   # Feature development
├── feature/data-pipeline   # Another feature
├── hotfix/security-fix     # Critical fixes
└── release/v1.2.0         # Release preparation

# Branch naming conventions:
# feature/ISSUE-brief-description
# bugfix/ISSUE-brief-description
# hotfix/ISSUE-brief-description
# release/version-number
```

## 8. 📋 **CODE QUALITY CHECKLIST**

### **Before every commit:**
- [ ] **Code passes all linting** (black, flake8, mypy)
- [ ] **All tests pass** (unit, integration, e2e)
- [ ] **Test coverage ≥90%** for new code
- [ ] **No hardcoded values** or secrets
- [ ] **Type hints** on all public functions
- [ ] **Docstrings** for all public classes/methods
- [ ] **Error handling** implemented
- [ ] **Logging** added for important operations
- [ ] **Performance** impact considered
- [ ] **Security** vulnerabilities checked
- [ ] **Dependencies** properly declared
- [ ] **Documentation** updated if needed

### **Before every Pull Request:**
- [ ] **Branch is up to date** with main/develop
- [ ] **PR description** explains changes clearly
- [ ] **Breaking changes** documented
- [ ] **Migration guide** provided if needed
- [ ] **Reviewers assigned** appropriately
- [ ] **Labels and milestones** set
- [ ] **Related issues** linked
- [ ] **Screenshots/demos** for UI changes

### **AI/ML Specific Checks:**
- [ ] **No data leakage** in pipelines
- [ ] **Reproducible results** (random seeds set)
- [ ] **Model artifacts** properly versioned
- [ ] **Experiment tracking** implemented
- [ ] **Data validation** in place
- [ ] **Resource usage** optimized
- [ ] **Bias and fairness** considered
- [ ] **Model interpretability** addressed

## 9. 🚫 **COMMON PITFALLS TO AVOID**

### 9.1 **Code Smells**
```python
# ❌ ANTI-PATTERN: Long parameter lists
def bad_function(param1, param2, param3, param4, param5, param6, param7, param8):
    pass

# ✅ CORRECT: Use configuration objects
@dataclass
class ProcessingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 10

def good_function(data: np.ndarray, config: ProcessingConfig):
    pass

# ❌ ANTI-PATTERN: Deep nesting
def bad_nested_function(data):
    if data:
        if len(data) > 0:
            if isinstance(data[0], dict):
                if 'key' in data[0]:
                    if data[0]['key'] is not None:
                        return process_data(data)
    return None

# ✅ CORRECT: Early returns and guard clauses
def good_function(data):
    if not data or len(data) == 0:
        return None
    
    if not isinstance(data[0], dict):
        return None
    
    if 'key' not in data[0] or data[0]['key'] is None:
        return None
    
    return process_data(data)
```

## 10. 🎯 **TEAM COLLABORATION STANDARDS**

### 10.1 **Code Review Guidelines**
- **Review within 24 hours** of PR creation
- **Focus on logic, not style** (automated tools handle style)
- **Provide constructive feedback** with examples
- **Approve only when confident** in code quality
- **Test changes locally** when needed
- **Check for security implications**
- **Verify documentation updates**

### 10.2 **Communication Standards**
- **Use issue templates** for bug reports and features
- **Tag relevant team members** in discussions
- **Document decisions** in issues or wiki
- **Share knowledge** through code comments and docs
- **Conduct regular code walkthroughs**
- **Maintain team coding standards** document

### 10.3 **Release Process**
- **Semantic versioning** (MAJOR.MINOR.PATCH)
- **Changelog maintenance** for each release
- **Release notes** with migration guides
- **Backward compatibility** consideration
- **Deprecation warnings** before breaking changes
- **Security updates** prioritized and fast-tracked




---
description: 
globs: 
alwaysApply: true
---
---
description: Comprehensive guidelines for automated project tracking, CSV logging, and Jira integration for AI/ML development workflows.
globs: logs/**/*.csv, jira/**/*.csv, tracking/**/*.csv, reports/**/*.csv, .github/workflows/*.yml
alwaysApply: true
---

# 📊 AUTOMATED PROJECT TRACKING & JIRA INTEGRATION FOR AI/ML

## 1. 🏗️ **COMPREHENSIVE CSV LOGGING STRUCTURE**

### 1.1 **Enhanced Jira-Compatible Fields**
```csv
Issue Key,Project Key,Summary,Description,Issue Type,Priority,Status,Resolution,
Assignee,Reporter,Created Date,Updated Date,Resolved Date,Due Date,
Time Original Estimate,Time Spent,Time Remaining,Story Points,Epic Link,
Sprint,Labels,Components,Fix Version,Affects Version,Environment,
AI Model Type,Dataset Size,Algorithm Used,Performance Metric,Baseline Score,
Achieved Score,Data Source,Feature Count,Training Time,Inference Time,
Memory Usage,GPU Utilization,Code Coverage,Test Status,Deployment Stage,
Git Commit Hash,Branch Name,Pull Request,Code Review Status,
Business Value,Technical Debt,Risk Level,Compliance Status
```

### 1.2 **AI/ML Specific Tracking Fields**
```python
# ✅ EXCELLENT: Structured AI/ML tracking data
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import csv

@dataclass
class AIMLIssueTracker:
    """Comprehensive tracking for AI/ML development issues."""
    
    # Core Jira Fields
    issue_key: str
    project_key: str = "AIML"
    summary: str = ""
    description: str = ""
    issue_type: str = "Story"  # Story, Bug, Epic, Task, Sub-task
    priority: str = "Medium"   # Highest, High, Medium, Low, Lowest
    status: str = "To Do"      # To Do, In Progress, Code Review, Testing, Done
    assignee: str = ""
    reporter: str = ""
    
    # Time Tracking
    created_date: str = ""
    updated_date: str = ""
    resolved_date: Optional[str] = None
    due_date: Optional[str] = None
    time_estimate: str = "0h"     # Original estimate
    time_spent: str = "0h"        # Actual time spent
    time_remaining: str = "0h"    # Remaining estimate
    story_points: int = 0
    
    # Project Organization
    epic_link: Optional[str] = None
    sprint: Optional[str] = None
    labels: List[str] = None
    components: List[str] = None
    fix_version: Optional[str] = None
    
    # AI/ML Specific Fields
    ai_model_type: str = ""           # RandomForest, NeuralNetwork, etc.
    dataset_size: int = 0             # Number of samples
    algorithm_used: str = ""          # Specific algorithm implementation
    performance_metric: str = ""      # Accuracy, F1, RMSE, etc.
    baseline_score: Optional[float] = None
    achieved_score: Optional[float] = None
    data_source: str = ""             # Source of training data
    feature_count: int = 0            # Number of features
    training_time: str = "0h"         # Time to train model
    inference_time_ms: float = 0.0    # Average inference time
    memory_usage_mb: float = 0.0      # Peak memory usage
    gpu_utilization: float = 0.0      # GPU utilization percentage
    
    # Code Quality
    code_coverage: float = 0.0        # Test coverage percentage
    test_status: str = "Pending"      # Passed, Failed, Pending
    deployment_stage: str = "Dev"     # Dev, Staging, Production
    
    # Version Control
    git_commit_hash: str = ""
    branch_name: str = ""
    pull_request: str = ""
    code_review_status: str = "Pending"  # Approved, Changes Requested, Pending
    
    # Business & Risk
    business_value: str = "Medium"    # High, Medium, Low
    technical_debt: str = "None"      # High, Medium, Low, None
    risk_level: str = "Low"           # High, Medium, Low
    compliance_status: str = "Compliant"  # Compliant, Non-compliant, Under Review
    
    def __post_init__(self):
        """Initialize default values and validate data."""
        if not self.created_date:
            self.created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.updated_date:
            self.updated_date = self.created_date
        if self.labels is None:
            self.labels = []
        if self.components is None:
            self.components = []
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to CSV-compatible dictionary."""
        data = asdict(self)
        # Convert lists to comma-separated strings for CSV
        data['labels'] = ','.join(self.labels) if self.labels else ""
        data['components'] = ','.join(self.components) if self.components else ""
        return data
    
    def update_time_tracking(self, time_spent_delta: str) -> None:
        """Update time tracking information."""
        self.time_spent = self._add_time(self.time_spent, time_spent_delta)
        self.updated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_ai_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update AI/ML specific metrics."""
        self.performance_metric = metrics.get('metric_type', self.performance_metric)
        self.achieved_score = metrics.get('score', self.achieved_score)
        self.training_time = metrics.get('training_time', self.training_time)
        self.inference_time_ms = metrics.get('inference_time_ms', self.inference_time_ms)
        self.memory_usage_mb = metrics.get('memory_mb', self.memory_usage_mb)
        self.updated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## 2. 🤖 **AUTOMATED TRACKING INTEGRATION**

### 2.1 **Git Hooks for Automatic Logging**
```bash
#!/bin/bash
# .git/hooks/post-commit
# Automatic CSV logging on commit

COMMIT_HASH=$(git rev-parse HEAD)
BRANCH_NAME=$(git branch --show-current)
COMMIT_MESSAGE=$(git log -1 --pretty=%B)
AUTHOR=$(git log -1 --pretty=format:'%an')
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Extract Jira issue key from commit message
ISSUE_KEY=$(echo "$COMMIT_MESSAGE" | grep -oE '[A-Z]+-[0-9]+' | head -1)

if [ ! -z "$ISSUE_KEY" ]; then
    # Update CSV log with commit information
    python scripts/update_tracking_log.py \
        --issue-key "$ISSUE_KEY" \
        --commit-hash "$COMMIT_HASH" \
        --branch-name "$BRANCH_NAME" \
        --author "$AUTHOR" \
        --message "$COMMIT_MESSAGE"
fi
```

### 2.2 **GitHub Actions Integration**
```yaml
# .github/workflows/project-tracking.yml
name: Automated Project Tracking

on:
  push:
    branches: [ main, develop, 'feature/*' ]
  pull_request:
    branches: [ main, develop ]
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types:
      - completed

jobs:
  update-tracking:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install pandas requests python-dotenv jira
          
      - name: Extract commit information
        id: commit-info
        run: |
          echo "commit_hash=${{ github.sha }}" >> $GITHUB_OUTPUT
          echo "branch_name=${{ github.ref_name }}" >> $GITHUB_OUTPUT
          echo "author=${{ github.actor }}" >> $GITHUB_OUTPUT
          
      - name: Update project tracking
        env:
          JIRA_URL: ${{ secrets.JIRA_URL }}
          JIRA_USERNAME: ${{ secrets.JIRA_USERNAME }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
        run: |
          python scripts/automated_tracking.py \
            --commit-hash "${{ steps.commit-info.outputs.commit_hash }}" \
            --branch-name "${{ steps.commit-info.outputs.branch_name }}" \
            --author "${{ steps.commit-info.outputs.author }}" \
            --action "${{ github.event_name }}"
            
      - name: Update test results
        if: github.event_name == 'workflow_run'
        run: |
          python scripts/update_test_metrics.py \
            --workflow-result "${{ github.event.workflow_run.conclusion }}"
            
      - name: Generate reports
        run: |
          python scripts/generate_tracking_reports.py \
            --output-dir reports/ \
            --format both  # CSV and HTML
            
      - name: Upload tracking artifacts
        uses: actions/upload-artifact@v3
        with:
          name: project-tracking-reports
          path: reports/
```

### 2.3 **Automated Tracking Script**
```python
# scripts/automated_tracking.py
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import re
import subprocess
import json

class AutomatedProjectTracker:
    """Automated project tracking for AI/ML development."""
    
    def __init__(self, csv_path: str = "logs/project_tracking.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not self.csv_path.exists():
            self._initialize_csv()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV with proper headers."""
        headers = [
            "Issue Key", "Project Key", "Summary", "Description", "Issue Type",
            "Priority", "Status", "Assignee", "Reporter", "Created Date",
            "Updated Date", "Time Spent", "Story Points", "Epic Link",
            "Sprint", "Labels", "AI Model Type", "Performance Metric",
            "Achieved Score", "Git Commit Hash", "Branch Name", "Code Coverage",
            "Test Status", "Business Value", "Risk Level"
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.csv_path, index=False)
    
    def extract_issue_from_commit(self, commit_message: str) -> Optional[str]:
        """Extract Jira issue key from commit message."""
        # Match patterns like AIML-123, PROJ-456, etc.
        pattern = r'([A-Z]+-\d+)'
        match = re.search(pattern, commit_message)
        return match.group(1) if match else None
    
    def get_commit_info(self, commit_hash: str) -> Dict[str, Any]:
        """Get detailed commit information."""
        try:
            # Get commit details
            commit_info = subprocess.run([
                'git', 'show', '--format=%an|%ae|%ad|%s|%b', '--no-patch', commit_hash
            ], capture_output=True, text=True, check=True)
            
            lines = commit_info.stdout.strip().split('|')
            
            # Get file changes
            file_changes = subprocess.run([
                'git', 'diff-tree', '--no-commit-id', '--name-status', '-r', commit_hash
            ], capture_output=True, text=True, check=True)
            
            return {
                'author': lines[0] if len(lines) > 0 else '',
                'email': lines[1] if len(lines) > 1 else '',
                'date': lines[2] if len(lines) > 2 else '',
                'subject': lines[3] if len(lines) > 3 else '',
                'body': lines[4] if len(lines) > 4 else '',
                'files_changed': file_changes.stdout.strip().split('\n') if file_changes.stdout.strip() else []
            }
        except subprocess.CalledProcessError:
            return {}
    
    def analyze_code_changes(self, files_changed: list) -> Dict[str, Any]:
        """Analyze the type and scope of code changes."""
        analysis = {
            'change_type': 'Other',
            'ai_related': False,
            'test_files': 0,
            'source_files': 0,
            'config_files': 0,
            'model_files': 0
        }
        
        for file_change in files_changed:
            if not file_change:
                continue
                
            parts = file_change.split('\t')
            if len(parts) < 2:
                continue
                
            status, filename = parts[0], parts[1]
            
            if filename.endswith('.py'):
                if '/test' in filename or filename.startswith('test_'):
                    analysis['test_files'] += 1
                elif '/models/' in filename or 'model' in filename.lower():
                    analysis['model_files'] += 1
                    analysis['ai_related'] = True
                else:
                    analysis['source_files'] += 1
                    
            elif filename.endswith(('.yml', '.yaml', '.json', '.toml')):
                analysis['config_files'] += 1
        
        # Determine change type
        if analysis['model_files'] > 0:
            analysis['change_type'] = 'AI/ML Model'
        elif analysis['test_files'] > 0:
            analysis['change_type'] = 'Testing'
        elif analysis['config_files'] > 0:
            analysis['change_type'] = 'Configuration'
        elif analysis['source_files'] > 0:
            analysis['change_type'] = 'Feature/Bug Fix'
        
        return analysis
    
    def update_tracking(
        self, 
        commit_hash: str, 
        branch_name: str, 
        author: str,
        action: str = "commit"
    ) -> None:
        """Update tracking with commit information."""
        
        commit_info = self.get_commit_info(commit_hash)
        if not commit_info:
            return
        
        issue_key = self.extract_issue_from_commit(commit_info.get('subject', ''))
        if not issue_key:
            # Create a generic entry for commits without issue keys
            issue_key = f"AUTO-{commit_hash[:8]}"
        
        analysis = self.analyze_code_changes(commit_info.get('files_changed', []))
        
        # Load existing data
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        
        # Check if issue already exists
        if issue_key in df['Issue Key'].values:
            # Update existing issue
            idx = df[df['Issue Key'] == issue_key].index[0]
            df.at[idx, 'Updated Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.at[idx, 'Git Commit Hash'] = commit_hash
            df.at[idx, 'Branch Name'] = branch_name
            
            # Update status based on branch and action
            if action == "pull_request" and "main" in branch_name:
                df.at[idx, 'Status'] = "Code Review"
            elif action == "push" and branch_name == "main":
                df.at[idx, 'Status'] = "Done"
                
        else:
            # Create new issue entry
            new_row = {
                'Issue Key': issue_key,
                'Project Key': 'AIML',
                'Summary': commit_info.get('subject', '')[:100],
                'Description': commit_info.get('body', '')[:500],
                'Issue Type': 'Task' if analysis['change_type'] == 'AI/ML Model' else 'Story',
                'Priority': 'High' if analysis['ai_related'] else 'Medium',
                'Status': 'In Progress',
                'Assignee': author,
                'Reporter': author,
                'Created Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Updated Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Time Spent': '1h',  # Default estimate
                'Story Points': 3 if analysis['ai_related'] else 2,
                'Epic Link': 'AIML-EPIC-1' if analysis['ai_related'] else '',
                'Labels': 'ai-ml' if analysis['ai_related'] else 'development',
                'AI Model Type': 'Machine Learning' if analysis['ai_related'] else '',
                'Git Commit Hash': commit_hash,
                'Branch Name': branch_name,
                'Business Value': 'High' if analysis['ai_related'] else 'Medium',
                'Risk Level': 'Medium' if analysis['ai_related'] else 'Low'
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save updated data
        df.to_csv(self.csv_path, index=False)
        print(f"Updated tracking for {issue_key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated project tracking")
    parser.add_argument("--commit-hash", required=True)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--action", default="commit")
    
    args = parser.parse_args()
    
    tracker = AutomatedProjectTracker()
    tracker.update_tracking(
        args.commit_hash,
        args.branch_name,
        args.author,
        args.action
    )
```

## 3. 📈 **ADVANCED REPORTING & ANALYTICS**

### 3.1 **Automated Report Generation**
```python
# scripts/generate_tracking_reports.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, Any, List

class ProjectAnalytics:
    """Generate comprehensive project analytics and reports."""
    
    def __init__(self, csv_path: str = "logs/project_tracking.csv"):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        
    def generate_velocity_report(self) -> Dict[str, Any]:
        """Generate team velocity and productivity metrics."""
        if self.df.empty:
            return {}
        
        # Convert dates
        self.df['Created Date'] = pd.to_datetime(self.df['Created Date'])
        self.df['Updated Date'] = pd.to_datetime(self.df['Updated Date'])
        
        # Calculate velocity by week
        self.df['Week'] = self.df['Created Date'].dt.isocalendar().week
        velocity_by_week = self.df.groupby('Week')['Story Points'].sum()
        
        # Completion rate
        completed_issues = len(self.df[self.df['Status'] == 'Done'])
        total_issues = len(self.df)
        completion_rate = (completed_issues / total_issues * 100) if total_issues > 0 else 0
        
        return {
            'velocity_by_week': velocity_by_week.to_dict(),
            'completion_rate': completion_rate,
            'total_story_points': self.df['Story Points'].sum(),
            'completed_story_points': self.df[self.df['Status'] == 'Done']['Story Points'].sum(),
            'average_cycle_time': self._calculate_average_cycle_time()
        }
    
    def generate_ai_ml_metrics(self) -> Dict[str, Any]:
        """Generate AI/ML specific project metrics."""
        ai_issues = self.df[self.df['AI Model Type'].notna() & (self.df['AI Model Type'] != '')]
        
        if ai_issues.empty:
            return {}
        
        return {
            'total_ai_issues': len(ai_issues),
            'completed_ai_issues': len(ai_issues[ai_issues['Status'] == 'Done']),
            'average_model_score': ai_issues['Achieved Score'].mean(),
            'model_types_distribution': ai_issues['AI Model Type'].value_counts().to_dict(),
            'performance_trends': self._calculate_performance_trends(ai_issues)
        }
    
    def generate_risk_analysis(self) -> Dict[str, Any]:
        """Generate risk analysis report."""
        return {
            'high_risk_issues': len(self.df[self.df['Risk Level'] == 'High']),
            'technical_debt_issues': len(self.df[self.df['Technical Debt'].isin(['High', 'Medium'])]),
            'overdue_issues': self._count_overdue_issues(),
            'blocked_issues': len(self.df[self.df['Status'] == 'Blocked']),
            'risk_distribution': self.df['Risk Level'].value_counts().to_dict()
        }
    
    def export_jira_import_csv(self, output_path: str) -> None:
        """Export data in Jira import format."""
        if self.df.empty:
            return
        
        # Jira import format
        jira_columns = [
            'Issue Key', 'Summary', 'Description', 'Issue Type', 'Priority',
            'Status', 'Assignee', 'Reporter', 'Created', 'Updated',
            'Time Original Estimate', 'Time Spent', 'Story Points',
            'Epic Link', 'Sprint', 'Labels', 'Components'
        ]
        
        export_df = self.df.copy()
        
        # Map our columns to Jira columns
        column_mapping = {
            'Created Date': 'Created',
            'Updated Date': 'Updated',
            'Time Estimate': 'Time Original Estimate'
        }
        
        export_df = export_df.rename(columns=column_mapping)
        
        # Filter to Jira columns that exist in our data
        available_columns = [col for col in jira_columns if col in export_df.columns]
        export_df = export_df[available_columns]
        
        export_df.to_csv(output_path, index=False)
        print(f"Jira import file exported to: {output_path}")
```

## 4. 🔄 **WORKFLOW AUTOMATION**

### 4.1 **Jira API Integration**
```python
# scripts/jira_integration.py
from jira import JIRA
import pandas as pd
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

class JiraAutomation:
    """Automated Jira integration for AI/ML projects."""
    
    def __init__(self):
        self.jira_url = os.getenv('JIRA_URL')
        self.username = os.getenv('JIRA_USERNAME')
        self.api_token = os.getenv('JIRA_API_TOKEN')
        self.project_key = os.getenv('JIRA_PROJECT_KEY', 'AIML')
        
        if not all([self.jira_url, self.username, self.api_token]):
            raise ValueError("Missing required Jira credentials in environment variables")
        
        self.jira = JIRA(
            server=self.jira_url,
            basic_auth=(self.username, self.api_token)
        )
    
    def create_ai_ml_issue(self, issue_data: Dict[str, Any]) -> str:
        """Create AI/ML specific issue in Jira."""
        
        # AI/ML specific custom fields
        custom_fields = {
            'customfield_10001': issue_data.get('ai_model_type', ''),      # AI Model Type
            'customfield_10002': issue_data.get('dataset_size', 0),       # Dataset Size
            'customfield_10003': issue_data.get('performance_metric', ''), # Performance Metric
            'customfield_10004': issue_data.get('achieved_score', 0.0),   # Achieved Score
            'customfield_10005': issue_data.get('baseline_score', 0.0),   # Baseline Score
        }
        
        issue_dict = {
            'project': {'key': self.project_key},
            'summary': issue_data['summary'],
            'description': issue_data.get('description', ''),
            'issuetype': {'name': issue_data.get('issue_type', 'Story')},
            'priority': {'name': issue_data.get('priority', 'Medium')},
            'assignee': {'name': issue_data.get('assignee', '')},
            'labels': issue_data.get('labels', []),
            **custom_fields
        }
        
        # Add epic link if provided
        if issue_data.get('epic_link'):
            issue_dict['customfield_10014'] = issue_data['epic_link']  # Epic Link field
        
        new_issue = self.jira.create_issue(fields=issue_dict)
        
        # Add time tracking if provided
        if issue_data.get('time_estimate'):
            self.jira.add_worklog(
                new_issue,
                timeSpent=issue_data['time_estimate'],
                comment="Initial time estimate"
            )
        
        return new_issue.key
    
    def sync_csv_to_jira(self, csv_path: str) -> List[str]:
        """Sync CSV data to Jira issues."""
        df = pd.read_csv(csv_path)
        created_issues = []
        
        for _, row in df.iterrows():
            try:
                # Check if issue already exists
                if pd.notna(row.get('Issue Key', '')) and row['Issue Key'].startswith(self.project_key):
                    # Update existing issue
                    self._update_existing_issue(row)
                else:
                    # Create new issue
                    issue_key = self.create_ai_ml_issue(row.to_dict())
                    created_issues.append(issue_key)
                    
            except Exception as e:
                print(f"Error processing row {row.get('Summary', 'Unknown')}: {str(e)}")
        
        return created_issues
    
    def generate_sprint_report(self, sprint_id: str) -> Dict[str, Any]:
        """Generate comprehensive sprint report."""
        sprint_issues = self.jira.search_issues(
            f'project = {self.project_key} AND sprint = {sprint_id}',
            expand='changelog'
        )
        
        report = {
            'sprint_id': sprint_id,
            'total_issues': len(sprint_issues),
            'completed_issues': 0,
            'total_story_points': 0,
            'completed_story_points': 0,
            'ai_ml_issues': 0,
            'bug_count': 0,
            'story_count': 0,
            'task_count': 0,
            'cycle_times': [],
            'performance_improvements': []
        }
        
        for issue in sprint_issues:
            # Basic metrics
            if issue.fields.status.name in ['Done', 'Closed']:
                report['completed_issues'] += 1
                if hasattr(issue.fields, 'customfield_10016'):  # Story Points
                    points = getattr(issue.fields, 'customfield_10016', 0) or 0
                    report['completed_story_points'] += points
            
            # Issue type counts
            issue_type = issue.fields.issuetype.name
            if issue_type == 'Bug':
                report['bug_count'] += 1
            elif issue_type == 'Story':
                report['story_count'] += 1
            elif issue_type == 'Task':
                report['task_count'] += 1
            
            # AI/ML specific tracking
            if hasattr(issue.fields, 'customfield_10001'):  # AI Model Type
                ai_model_type = getattr(issue.fields, 'customfield_10001', '')
                if ai_model_type:
                    report['ai_ml_issues'] += 1
                    
                    # Track performance improvements
                    baseline = getattr(issue.fields, 'customfield_10005', 0) or 0
                    achieved = getattr(issue.fields, 'customfield_10004', 0) or 0
                    if baseline > 0 and achieved > baseline:
                        improvement = ((achieved - baseline) / baseline) * 100
                        report['performance_improvements'].append({
                            'issue_key': issue.key,
                            'model_type': ai_model_type,
                            'improvement_percent': improvement
                        })
        
        return report
```

## 5. 📊 **COMPLIANCE & AUDIT TRACKING**

### 5.1 **Audit Trail Management**
```python
# scripts/audit_compliance.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import hashlib
import json

class ComplianceAuditor:
    """Audit and compliance tracking for AI/ML projects."""
    
    def __init__(self, csv_path: str = "logs/project_tracking.csv"):
        self.csv_path = csv_path
        self.audit_log_path = "logs/audit_trail.json"
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        df = pd.read_csv(self.csv_path)
        
        compliance_metrics = {
            'data_governance': self._check_data_governance(df),
            'model_documentation': self._check_model_documentation(df),
            'code_review_compliance': self._check_code_review_compliance(df),
            'testing_compliance': self._check_testing_compliance(df),
            'security_compliance': self._check_security_compliance(df),
            'audit_trail_integrity': self._verify_audit_trail()
        }
        
        # Calculate overall compliance score
        compliance_scores = [metric.get('score', 0) for metric in compliance_metrics.values()]
        overall_score = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
        
        return {
            'overall_compliance_score': overall_score,
            'compliance_grade': self._get_compliance_grade(overall_score),
            'detailed_metrics': compliance_metrics,
            'recommendations': self._generate_compliance_recommendations(compliance_metrics),
            'report_date': datetime.now().isoformat()
        }
    
    def _check_data_governance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data governance compliance."""
        ai_issues = df[df['AI Model Type'].notna() & (df['AI Model Type'] != '')]
        
        if ai_issues.empty:
            return {'score': 100, 'status': 'N/A', 'details': 'No AI/ML issues to evaluate'}
        
        # Check for required data source documentation
        documented_sources = ai_issues['Data Source'].notna() & (ai_issues['Data Source'] != '')
        documentation_rate = documented_sources.sum() / len(ai_issues) * 100
        
        return {
            'score': documentation_rate,
            'status': 'Compliant' if documentation_rate >= 90 else 'Non-compliant',
            'details': f'{documentation_rate:.1f}% of AI/ML issues have documented data sources',
            'missing_documentation': len(ai_issues) - documented_sources.sum()
        }
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]) -> str:
        """Log audit event with tamper-proof hash."""
        timestamp = datetime.now().isoformat()
        event_id = hashlib.sha256(f"{timestamp}{event_type}{json.dumps(details, sort_keys=True)}".encode()).hexdigest()[:16]
        
        audit_entry = {
            'event_id': event_id,
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details,
            'user': details.get('user', 'system'),
            'hash': self._calculate_entry_hash(event_id, timestamp, event_type, details)
        }
        
        # Append to audit log
        try:
            with open(self.audit_log_path, 'r') as f:
                audit_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            audit_log = []
        
        audit_log.append(audit_entry)
        
        with open(self.audit_log_path, 'w') as f:
            json.dump(audit_log, f, indent=2)
        
        return event_id
```

## 6. 📋 **IMPLEMENTATION CHECKLIST**

### **Setup Requirements:**
- [ ] **Jira instance configured** with AI/ML custom fields
- [ ] **GitHub repository** with proper webhook integration
- [ ] **CSV logging directory** created with proper permissions
- [ ] **Environment variables** set for Jira API access
- [ ] **Git hooks** installed for automatic tracking
- [ ] **CI/CD pipeline** configured with tracking steps

### **Daily Operations:**
- [ ] **Commit messages** include Jira issue keys
- [ ] **Pull requests** linked to relevant issues
- [ ] **Time tracking** updated accurately
- [ ] **AI/ML metrics** recorded for model changes
- [ ] **Test results** automatically logged
- [ ] **Code coverage** tracked and reported

### **Weekly Reviews:**
- [ ] **Velocity reports** generated and reviewed
- [ ] **Risk assessment** updated
- [ ] **Compliance metrics** checked
- [ ] **Technical debt** evaluated
- [ ] **Performance trends** analyzed
- [ ] **Sprint planning** based on data insights

### **Monthly Audits:**
- [ ] **Audit trail integrity** verified
- [ ] **Compliance report** generated
- [ ] **Data governance** reviewed
- [ ] **Security assessment** completed
- [ ] **Process improvements** identified
- [ ] **Stakeholder reports** distributed

## 7. 🎯 **BEST PRACTICES SUMMARY**

### **DO:**
- ✅ **Automate tracking** wherever possible
- ✅ **Include Jira keys** in commit messages
- ✅ **Track AI/ML metrics** consistently
- ✅ **Document data sources** and lineage
- ✅ **Maintain audit trails** for compliance
- ✅ **Generate regular reports** for stakeholders
- ✅ **Use structured data** formats
- ✅ **Integrate with CI/CD** pipelines

### **DON'T:**
- ❌ **Manual data entry** without validation
- ❌ **Skip time tracking** for accurate planning
- ❌ **Ignore compliance** requirements
- ❌ **Mix personal and project** tracking
- ❌ **Forget to backup** tracking data
- ❌ **Use inconsistent** field formats
- ❌ **Skip security** considerations
- ❌ **Ignore data quality** issues

## 8. 📊 **EXAMPLE USAGE**

### **Sample CSV Entry:**
```csv
Issue Key,Project Key,Summary,Description,Issue Type,Priority,Status,Assignee,Created Date,AI Model Type,Performance Metric,Achieved Score,Git Commit Hash
AIML-101,AIML,"Implement Random Forest classifier","Added Random Forest model for user classification with 95% accuracy",Story,High,Done,john.doe,2024-01-15 10:30:00,RandomForest,Accuracy,0.95,a1b2c3d4e5f6
AIML-102,AIML,"Optimize model inference speed","Reduced inference time from 100ms to 20ms using batch processing",Task,Medium,In Progress,jane.smith,2024-01-16 14:20:00,NeuralNetwork,Inference Time,20.0,f6e5d4c3b2a1
```

### **Automated Report Output:**
```json
{
  "velocity_report": {
    "completion_rate": 85.5,
    "total_story_points": 156,
    "completed_story_points": 134
  },
  "ai_ml_metrics": {
    "total_ai_issues": 25,
    "completed_ai_issues": 20,
    "average_model_score": 0.89,
    "model_types": {
      "RandomForest": 12,
      "NeuralNetwork": 8,
      "SVM": 5
    }
  }
}
```


---
description: 
globs: 
alwaysApply: true
---
---
description: Comprehensive Scrum and Agile management framework for AI/ML projects with specialized ceremonies, metrics, and best practices.
globs: tasks/**/*.md, jira/**/*.md, sprints/**/*.md, retrospectives/**/*.md, planning/**/*.md
alwaysApply: true
---

# 🏃‍♂️ SCRUM & AGILE MANAGEMENT FOR AI/ML PROJECTS

## 1. 🎯 **AI/ML SCRUM ROLES & STRUCTURE**

### 1.1 **Specialized Roles**
```markdown
🔹 **Product Owner (AI/ML Focus)**
- Define AI/ML product vision and strategy
- Manage AI/ML feature backlog and priorities
- Define acceptance criteria for model performance
- Stakeholder communication for AI/ML outcomes

🔹 **Scrum Master (AI/ML Specialized)**
- Facilitate AI/ML specific ceremonies
- Remove impediments related to data and infrastructure
- Coach team on AI/ML best practices
- Ensure compliance with AI/ML governance

🔹 **Development Team (Cross-functional)**
- Data Scientists & ML Engineers
- Software Engineers & DevOps Engineers
- Data Engineers & Analytics Engineers
- QA Engineers with ML testing expertise

🔹 **Additional AI/ML Roles**
- Data Steward (data quality and governance)
- MLOps Engineer (model deployment and monitoring)
- AI Ethics Officer (bias and fairness evaluation)
```

### 1.2 **AI/ML Sprint Structure (2 weeks)**
```markdown
**Week 1: Research & Development**
Day 1-2: Sprint Planning & Data Analysis
Day 3-5: Model Development & Experimentation
Day 6-7: Initial Model Validation

**Week 2: Validation & Integration**
Day 8-9: Model Testing & Performance Evaluation
Day 10-11: Integration & Documentation
Day 12-14: Sprint Review & Retrospective
```

## 2. 📋 **AI/ML BACKLOG MANAGEMENT**

### 2.1 **User Story Template**
```markdown
🤖 **AI/ML Model Story:**
As a [user role]
I want [AI/ML capability]
So that [business value]

**Model Acceptance Criteria:**
- Model achieves minimum [metric] of [threshold]
- Inference time < [time limit]
- Bias metrics within acceptable range
- Model explainability requirements met

**Example:**
As a customer service manager
I want an automated sentiment analysis model
So that I can prioritize support tickets by urgency

**Acceptance Criteria:**
- Model achieves minimum F1-score of 0.85
- Inference time < 100ms per prediction
- Bias evaluation across demographics completed
```

### 2.2 **Prioritization Framework**
```python
# AI/ML Story Prioritization
def calculate_priority_score(story):
    """Calculate priority based on AI/ML factors."""
    factors = {
        'business_value': story.get('business_value', 5) * 0.3,
        'technical_risk': (10 - story.get('technical_risk', 5)) * 0.2,
        'data_availability': story.get('data_availability', 5) * 0.2,
        'model_complexity': (10 - story.get('model_complexity', 5)) * 0.15,
        'compliance_impact': story.get('compliance_impact', 5) * 0.15
    }
    return sum(factors.values())
```

## 3. 🗓️ **SPRINT PLANNING & EXECUTION**

### 3.1 **Sprint Planning Template**
```markdown
# Sprint [Number] Planning

## 🎯 Sprint Goal
[Clear, measurable AI/ML objective]

## 📊 Sprint Metrics
- **Velocity Target:** [Story Points]
- **Model Performance Target:** [Metric and threshold]
- **Code Coverage Target:** ≥90%

## 📋 Sprint Backlog Allocation
- 🔬 **Research & Experimentation:** 40% capacity
- 🛠️ **Development & Integration:** 35% capacity
- 🧪 **Testing & Validation:** 20% capacity
- 📚 **Documentation:** 5% capacity

## 🚧 Impediments & Dependencies
- [ ] Data access approval pending (Blocker)
- [ ] GPU resources allocation needed (High)
- [ ] Model review with stakeholders (Medium)
```

### 3.2 **Daily Standup Structure**
```markdown
🌅 **Daily Standup (15 minutes):**

**Each member answers:**
1. **Yesterday:** What AI/ML work did I complete?
2. **Today:** What AI/ML work will I focus on?
3. **Blockers:** What's preventing my progress?

**AI/ML Specific Updates:**
- Model training progress and results
- Data quality issues discovered
- Infrastructure constraints
- Experiment insights

**Example:**
👨‍💻 **Data Scientist:** Completed feature engineering, achieved 0.82 F1-score. Today: hyperparameter tuning. Blocker: Need larger GPU instance.
```

## 4. 📊 **AI/ML METRICS & TRACKING**

### 4.1 **Sprint Health Dashboard**
```python
from dataclasses import dataclass

@dataclass
class AIMLSprintMetrics:
    """AI/ML Sprint metrics tracker."""
    sprint_number: int
    velocity_planned: int
    velocity_achieved: int
    models_developed: int
    models_deployed: int
    experiment_success_rate: float
    code_coverage: float
    
    def calculate_health_score(self) -> float:
        """Calculate sprint health (0-100)."""
        velocity_score = min(100, (self.velocity_achieved / self.velocity_planned) * 100)
        quality_score = self.code_coverage
        ml_success_score = self.experiment_success_rate * 100
        
        return (velocity_score + quality_score + ml_success_score) / 3
```

### 4.2 **Velocity Tracking**
```markdown
| Sprint | Traditional | ML Dev | Data Eng | MLOps | Total |
|--------|-------------|--------|----------|-------|-------|
| 13     | 15 pts      | 8 pts  | 5 pts    | 3 pts | 31    |
| 14     | 18 pts      | 6 pts  | 7 pts    | 4 pts | 35    |
| 15     | 12 pts      | 10 pts | 4 pts    | 6 pts | 32    |

**Velocity Factors:**
- Data availability and quality
- Model experimentation cycles
- Infrastructure constraints
- Compliance requirements
```

## 5. 🔄 **AI/ML SCRUM CEREMONIES**

### 5.1 **Sprint Review (2 hours)**
```markdown
**Agenda:**
1. **Sprint Goals Review** (15 min): Performance targets achieved
2. **Live Demo** (45 min): Model functionality, metrics, pipelines
3. **Stakeholder Feedback** (30 min): Business input, technical questions
4. **Metrics Review** (15 min): Velocity, performance trends
5. **Next Sprint Preview** (15 min): Goals, priorities, dependencies

**Demo Checklist:**
- [ ] Model predictions working end-to-end
- [ ] Performance metrics displayed
- [ ] Data lineage demonstrated
- [ ] Error handling shown
- [ ] Monitoring functional
```

### 5.2 **Sprint Retrospective (1.5 hours)**
```markdown
**AI/ML Retrospective Structure:**

**Categories for Discussion:**
🔬 **Experimentation:** Design effectiveness, hypothesis validation
📊 **Data & Infrastructure:** Quality, availability, performance
🤖 **Model Development:** Workflows, deployment efficiency
👥 **Collaboration:** Cross-functional coordination, knowledge sharing

**Action Item Template:**
| Action | Owner | Timeline | Success Criteria |
|--------|-------|----------|------------------|
| Automated model validation | ML Engineer | Next Sprint | 100% models have tests |
| Data quality monitoring | Data Engineer | 2 Sprints | Zero production issues |
```

## 6. ✅ **DEFINITION OF DONE FOR AI/ML**

### 6.1 **Model Development DoD**
```markdown
**Code Quality & Testing**
- [ ] Code review completed and approved
- [ ] Unit tests written with ≥90% coverage
- [ ] Integration tests passing
- [ ] Code follows team standards

**Model Performance & Validation**
- [ ] Model meets minimum performance thresholds
- [ ] Cross-validation completed
- [ ] Bias and fairness evaluation completed
- [ ] Model interpretability requirements met

**Infrastructure & Deployment**
- [ ] Data lineage documented
- [ ] Model versioning configured
- [ ] Deployment pipeline tested
- [ ] Monitoring and alerting configured

**Documentation & Compliance**
- [ ] Model documentation updated
- [ ] API documentation current
- [ ] Compliance requirements verified
- [ ] Risk assessment completed
```

## 7. 🎯 **ESTIMATION & PLANNING**

### 7.1 **AI/ML Planning Poker**
```markdown
**Modified Fibonacci for AI/ML:**
- **1 pt:** Config change, minor bug fix
- **2 pts:** Data analysis, simple feature engineering
- **3 pts:** Baseline model implementation
- **5 pts:** Model experimentation, complex features
- **8 pts:** New architecture, infrastructure setup
- **13 pts:** End-to-end ML pipeline
- **21 pts:** Research spike, major architectural change

**Estimation Factors:**
- 🔍 Research uncertainty
- 📊 Data dependency and quality
- 🧪 Expected experimentation cycles
- 🏗️ Infrastructure complexity
- 👥 Cross-team dependencies
```

### 7.2 **Risk Management**
```markdown
**Risk Categories:**
🔴 **High Risk:** Data unavailability, model performance failure
🟡 **Medium Risk:** Team unavailability, requirement changes
🟢 **Low Risk:** Tool updates, performance optimization

**Risk Matrix:**
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Data pipeline failure | Medium | High | Automated backups |
| Model below target | Low | High | Multiple algorithms |
| Infrastructure scaling | High | Medium | Load testing |
```

## 8. 🛠️ **TOOLS & AUTOMATION**

### 8.1 **Tool Stack**
```yaml
project_management: "Jira with AI/ML custom fields"
collaboration: "Slack + Jira integration"
experiment_tracking: "MLflow"
model_registry: "MLflow Model Registry"
monitoring: "Evidently AI"
ci_cd: "GitHub Actions"
deployment: "Docker + Kubernetes"
```

### 8.2 **Automated Reporting**
```python
class AIMLScrumDashboard:
    """Automated Scrum reporting for AI/ML."""
    
    def generate_sprint_summary(self, sprint_id):
        return {
            'completion_rate': self._calculate_completion(),
            'velocity': self._get_velocity_data(),
            'ml_experiments': self._count_experiments(),
            'model_performance': self._get_best_scores(),
            'recommendations': self._generate_recommendations()
        }
```

## 9. 📈 **CONTINUOUS IMPROVEMENT**

### 9.1 **Team Maturity Assessment**
```markdown
**Maturity Dimensions (1-5 scale):**
1. **Agile Practices:** Scrum implementation quality
2. **ML Engineering:** Development best practices
3. **Data Management:** Governance and quality
4. **Collaboration:** Cross-functional effectiveness
5. **Automation:** CI/CD and MLOps maturity
6. **Measurement:** Metrics and monitoring
7. **Innovation:** Experimentation culture
8. **Compliance:** Governance and ethics
```

### 9.2 **Monthly Kaizen**
```markdown
**Improvement Session (2 hours):**
1. **Current State Analysis** (30 min): Review metrics
2. **Root Cause Analysis** (30 min): Identify problems
3. **Solution Generation** (45 min): Brainstorm improvements
4. **Action Planning** (15 min): Define specific actions

**Focus Areas:**
- 🚀 Process streamlining
- 🛠️ Tool enhancement
- 📚 Knowledge building
- 🤝 Collaboration improvement
- 📊 Better measurement
```

## 10. 📋 **IMPLEMENTATION CHECKLIST**

### **Week 1 - Setup:**
- [ ] Define AI/ML Scrum roles and responsibilities
- [ ] Configure project management tools
- [ ] Create initial AI/ML product backlog
- [ ] Establish Definition of Done for AI/ML
- [ ] Set up tracking and reporting automation

### **Week 2-4 - Process:**
- [ ] Conduct first sprint planning with AI/ML estimation
- [ ] Run daily standups with AI/ML updates
- [ ] Execute sprint review with model demonstrations
- [ ] Facilitate AI/ML focused retrospectives
- [ ] Implement continuous improvement feedback loops

### **Success Metrics:**
- [ ] Team velocity stable and predictable
- [ ] Sprint goals achieved >80% of time
- [ ] Stakeholder satisfaction >8/10
- [ ] Model delivery frequency increased
- [ ] Technical debt <20% of capacity

## 11. 🎯 **BEST PRACTICES SUMMARY**

### **DO:**
- ✅ Allocate 40% capacity for research and experimentation
- ✅ Include model performance targets in sprint goals
- ✅ Track AI/ML specific metrics alongside traditional ones
- ✅ Conduct bias and fairness evaluations regularly
- ✅ Maintain clear documentation for model decisions
- ✅ Implement automated testing for ML pipelines
- ✅ Plan for infrastructure and resource needs

### **DON'T:**
- ❌ Skip model validation and testing phases
- ❌ Ignore data quality and governance requirements
- ❌ Underestimate experimentation time needs
- ❌ Deploy models without proper monitoring
- ❌ Mix research and production development
- ❌ Forget stakeholder communication and feedback
- ❌ Neglect compliance and ethical considerations




---
description: 
globs: 
alwaysApply: true
---
---
description: Comprehensive AI/ML workflow guidelines with automated pipelines, model lifecycle management, and enterprise-grade development practices.
globs: workflows/**/*.md, pipelines/**/*.md, models/**/*.md, deployment/**/*.md, monitoring/**/*.md
alwaysApply: true
---

# 🔄 AI/ML WORKFLOW GUIDELINES & AUTOMATION FRAMEWORK

## 1. 🎯 **AI/ML WORKFLOW OVERVIEW**

### 1.1 **End-to-End ML Workflow**
```mermaid
flowchart TD
    A[Problem Definition] --> B[Data Discovery]
    B --> C[Data Preparation]
    C --> D[Feature Engineering]
    D --> E[Model Development]
    E --> F[Model Validation]
    F --> G[Model Deployment]
    G --> H[Model Monitoring]
    H --> I[Model Improvement]
    I --> E
    
    B --> J[Data Quality Gates]
    E --> K[Experiment Tracking]
    F --> L[A/B Testing]
    G --> M[CI/CD Pipeline]
    H --> N[Performance Alerts]
    
    style A fill:#e3f2fd
    style E fill:#e8f5e8
    style G fill:#fff3e0
    style H fill:#fce4ec
```

### 1.2 **Workflow Phases & Responsibilities**
```markdown
📊 **Phase 1: Data Discovery & Preparation (25% effort)**
- Data Scientists: EDA and data understanding
- Data Engineers: Pipeline design and implementation
- Domain Experts: Business context validation

🤖 **Phase 2: Model Development & Validation (35% effort)**
- ML Engineers: Model architecture and training
- Data Scientists: Feature engineering and selection
- QA Engineers: Model testing and validation

🚀 **Phase 3: Deployment & Integration (25% effort)**
- MLOps Engineers: Deployment pipeline setup
- Software Engineers: API integration
- DevOps Engineers: Infrastructure management

📈 **Phase 4: Monitoring & Improvement (15% effort)**
- Site Reliability Engineers: Production monitoring
- Data Scientists: Performance analysis
- Product Owners: Business impact assessment
```

## 2. 📋 **WORKFLOW AUTOMATION FRAMEWORK**

### 2.1 **GitHub Actions CI/CD Pipeline**
```yaml
# .github/workflows/ml_pipeline.yml
name: AI/ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  data_validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Data Quality
        run: python scripts/validate_data.py
      - name: Run Data Tests
        run: pytest tests/test_data.py

  model_training:
    needs: data_validation
    runs-on: ubuntu-latest
    steps:
      - name: Train Models
        run: python scripts/train_models.py
      - name: Log Experiments
        run: mlflow log-model --model-uri models/
      
  model_validation:
    needs: model_training
    runs-on: ubuntu-latest
    steps:
      - name: Validate Model Performance
        run: python scripts/validate_model.py
      - name: Bias Evaluation
        run: python scripts/bias_check.py
      
  deployment:
    needs: model_validation
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Staging
        run: docker build -t ml-model:latest .
      - name: Deploy to Production
        run: kubectl apply -f k8s/deployment.yaml
```

### 2.2 **Automated Workflow Orchestration**
```python
# workflows/ml_orchestrator.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import mlflow
import logging

class WorkflowStage(Enum):
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

@dataclass
class WorkflowStep:
    """Individual workflow step configuration."""
    name: str
    stage: WorkflowStage
    dependencies: List[str]
    timeout_minutes: int
    retry_count: int
    success_criteria: Dict[str, float]
    
class AIMLWorkflowOrchestrator:
    """Orchestrate AI/ML workflows with quality gates."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.current_run_id = None
        self.logger = logging.getLogger(__name__)
        
    def execute_workflow(self, workflow_name: str) -> Dict[str, any]:
        """Execute complete AI/ML workflow with gates."""
        with mlflow.start_run() as run:
            self.current_run_id = run.info.run_id
            self.logger.info(f"Starting workflow: {workflow_name}")
            
            results = {}
            workflow_steps = self.config['workflows'][workflow_name]
            
            for step in workflow_steps:
                try:
                    step_result = self._execute_step(step)
                    results[step.name] = step_result
                    
                    # Quality gate validation
                    if not self._validate_quality_gate(step, step_result):
                        raise WorkflowFailure(f"Quality gate failed for {step.name}")
                        
                    self.logger.info(f"Step {step.name} completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Step {step.name} failed: {str(e)}")
                    if step.retry_count > 0:
                        return self._retry_step(step)
                    else:
                        raise WorkflowFailure(f"Workflow failed at step: {step.name}")
            
            return {
                'status': 'success',
                'run_id': self.current_run_id,
                'results': results,
                'metrics': self._calculate_workflow_metrics(results)
            }
    
    def _execute_step(self, step: WorkflowStep) -> Dict[str, any]:
        """Execute individual workflow step."""
        start_time = time.time()
        
        # Execute step based on stage type
        if step.stage == WorkflowStage.DATA_VALIDATION:
            result = self._validate_data(step)
        elif step.stage == WorkflowStage.MODEL_TRAINING:
            result = self._train_model(step)
        elif step.stage == WorkflowStage.MODEL_VALIDATION:
            result = self._validate_model(step)
        elif step.stage == WorkflowStage.DEPLOYMENT:
            result = self._deploy_model(step)
        else:
            raise ValueError(f"Unknown workflow stage: {step.stage}")
        
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            f"{step.name}_duration": execution_time,
            f"{step.name}_success": 1.0
        })
        
        return result
    
    def _validate_quality_gate(self, step: WorkflowStep, result: Dict) -> bool:
        """Validate quality gates for workflow step."""
        for metric, threshold in step.success_criteria.items():
            if metric not in result:
                self.logger.warning(f"Missing metric {metric} for step {step.name}")
                return False
            
            if result[metric] < threshold:
                self.logger.error(f"Quality gate failed: {metric} = {result[metric]} < {threshold}")
                return False
        
        return True

# Example workflow configuration
workflow_config = {
    'workflows': {
        'customer_churn_prediction': [
            WorkflowStep(
                name="data_validation",
                stage=WorkflowStage.DATA_VALIDATION,
                dependencies=[],
                timeout_minutes=30,
                retry_count=2,
                success_criteria={
                    'data_quality_score': 0.95,
                    'completeness_ratio': 0.90
                }
            ),
            WorkflowStep(
                name="model_training",
                stage=WorkflowStage.MODEL_TRAINING,
                dependencies=["data_validation"],
                timeout_minutes=120,
                retry_count=1,
                success_criteria={
                    'accuracy': 0.85,
                    'f1_score': 0.80
                }
            )
        ]
    }
}
```

## 3. 🔍 **DATA WORKFLOW MANAGEMENT**

### 3.1 **Data Pipeline Workflow**
```markdown
📊 **Data Ingestion Workflow:**

**Step 1: Data Source Validation**
- [ ] Verify data source accessibility
- [ ] Check authentication and permissions
- [ ] Validate data schema compatibility
- [ ] Test connection stability

**Step 2: Data Quality Assessment**
- [ ] Run data profiling and statistics
- [ ] Identify missing values and outliers
- [ ] Validate data types and formats
- [ ] Check for data drift and anomalies

**Step 3: Data Transformation**
- [ ] Apply data cleaning rules
- [ ] Handle missing values and outliers
- [ ] Standardize formats and encodings
- [ ] Create derived features

**Step 4: Data Validation Gates**
- [ ] Completeness check (>90% required)
- [ ] Accuracy validation against business rules
- [ ] Consistency checks across data sources
- [ ] Timeliness validation (data freshness)

**Automated Data Quality Monitoring:**
```python
def validate_data_pipeline(data_path: str) -> Dict[str, float]:
    """Automated data quality validation."""
    df = pd.read_csv(data_path)
    
    quality_metrics = {
        'completeness': (df.count().sum() / df.size) * 100,
        'uniqueness': (df.nunique().sum() / len(df)) * 100,
        'validity': calculate_validity_score(df),
        'consistency': calculate_consistency_score(df)
    }
    
    # Quality gates
    gates_passed = all([
        quality_metrics['completeness'] >= 90,
        quality_metrics['validity'] >= 95,
        quality_metrics['consistency'] >= 98
    ])
    
    return {
        'metrics': quality_metrics,
        'gates_passed': gates_passed,
        'recommendations': generate_quality_recommendations(quality_metrics)
    }
```

### 3.2 **Feature Engineering Workflow**
```markdown
🔧 **Feature Engineering Pipeline:**

**Phase 1: Feature Discovery (20% effort)**
- Domain expert consultation
- Exploratory data analysis
- Feature importance analysis
- Business rule validation

**Phase 2: Feature Creation (40% effort)**
- Mathematical transformations
- Categorical encoding
- Time-based features
- Interaction features

**Phase 3: Feature Selection (25% effort)**
- Statistical significance testing
- Correlation analysis
- Feature importance ranking
- Dimensionality reduction

**Phase 4: Feature Validation (15% effort)**
- Cross-validation stability
- Performance impact assessment
- Business interpretation validation
- Documentation and cataloging
```

## 4. 🤖 **MODEL DEVELOPMENT WORKFLOW**

### 4.1 **Experiment Management Workflow**
```python
# workflows/experiment_manager.py
import mlflow
import mlflow.sklearn
from typing import Dict, List, Any

class ExperimentManager:
    """Manage ML experiments with automated tracking."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def run_experiment(self, 
                      model_config: Dict[str, Any],
                      data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML experiment with comprehensive tracking."""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(model_config)
            mlflow.log_params(data_config)
            
            # Train model
            model, metrics = self._train_model(model_config, data_config)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log artifacts
            self._log_experiment_artifacts(model, metrics)
            
            return {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'model_uri': f"runs:/{run.info.run_id}/model"
            }
    
    def compare_experiments(self, 
                           experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments and recommend best model."""
        
        experiments_data = []
        for exp_id in experiment_ids:
            run = mlflow.get_run(exp_id)
            experiments_data.append({
                'run_id': exp_id,
                'metrics': run.data.metrics,
                'params': run.data.params
            })
        
        # Find best model based on primary metric
        best_experiment = max(experiments_data, 
                            key=lambda x: x['metrics'].get('f1_score', 0))
        
        return {
            'best_run_id': best_experiment['run_id'],
            'best_metrics': best_experiment['metrics'],
            'comparison_report': self._generate_comparison_report(experiments_data)
        }

# Example usage
experiment_manager = ExperimentManager("customer_churn_detection")

model_configs = [
    {'algorithm': 'random_forest', 'n_estimators': 100, 'max_depth': 10},
    {'algorithm': 'gradient_boosting', 'n_estimators': 100, 'learning_rate': 0.1},
    {'algorithm': 'neural_network', 'hidden_layers': [64, 32], 'dropout': 0.2}
]

for config in model_configs:
    result = experiment_manager.run_experiment(config, data_config)
    print(f"Experiment {result['run_id']}: F1 = {result['metrics']['f1_score']:.3f}")
```

### 4.2 **Model Validation Workflow**
```markdown
✅ **Model Validation Pipeline:**

**Performance Validation (40% effort)**
- [ ] Cross-validation with 5+ folds
- [ ] Hold-out test set evaluation
- [ ] Time-series validation (if applicable)
- [ ] Performance stability across data segments

**Bias and Fairness Validation (25% effort)**
- [ ] Demographic parity assessment
- [ ] Equal opportunity evaluation
- [ ] Calibration analysis
- [ ] Disparate impact testing

**Business Validation (20% effort)**
- [ ] Business metric impact analysis
- [ ] Cost-benefit analysis
- [ ] ROI calculation
- [ ] Stakeholder approval process

**Technical Validation (15% effort)**
- [ ] Model interpretability assessment
- [ ] Feature importance analysis
- [ ] Model stability testing
- [ ] Adversarial robustness evaluation

**Validation Quality Gates:**
- Accuracy/F1 score meets minimum thresholds
- Bias metrics within acceptable ranges
- Business metrics show positive impact
- Technical requirements satisfied
```

## 5. 🚀 **DEPLOYMENT WORKFLOW**

### 5.1 **CI/CD Pipeline for ML Models**
```yaml
# deployment/ml_deployment_pipeline.yml
stages:
  - model_validation
  - staging_deployment
  - integration_testing
  - production_deployment
  - monitoring_setup

model_validation:
  script:
    - python validate_model_performance.py
    - python check_model_bias.py
    - python validate_model_size.py
  artifacts:
    reports:
      - validation_report.json
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

staging_deployment:
  script:
    - docker build -t ml-model:staging .
    - kubectl apply -f k8s/staging/
    - python run_smoke_tests.py
  environment:
    name: staging
    url: https://ml-api-staging.company.com
  
integration_testing:
  script:
    - python run_integration_tests.py
    - python load_testing.py
    - python a_b_testing_setup.py
  artifacts:
    reports:
      - integration_test_results.json

production_deployment:
  script:
    - docker build -t ml-model:production .
    - kubectl apply -f k8s/production/
    - python setup_monitoring.py
  environment:
    name: production
    url: https://ml-api.company.com
  when: manual
  only:
    - main
```

### 5.2 **Blue-Green Deployment Strategy**
```python
# deployment/blue_green_deployer.py
class BlueGreenMLDeployer:
    """Blue-Green deployment for ML models with traffic shifting."""
    
    def __init__(self, k8s_client, monitoring_client):
        self.k8s = k8s_client
        self.monitoring = monitoring_client
        
    def deploy_new_model(self, model_version: str) -> Dict[str, Any]:
        """Deploy new model version using blue-green strategy."""
        
        # Step 1: Deploy to green environment
        green_deployment = self._deploy_to_green(model_version)
        
        # Step 2: Run health checks
        health_status = self._run_health_checks(green_deployment)
        if not health_status['healthy']:
            self._rollback_green_deployment()
            raise DeploymentError("Health checks failed")
        
        # Step 3: Gradual traffic shifting
        traffic_shift_result = self._gradual_traffic_shift()
        
        # Step 4: Monitor performance
        performance_metrics = self._monitor_performance(duration_minutes=30)
        
        # Step 5: Complete switch or rollback
        if self._validate_deployment_success(performance_metrics):
            self._complete_blue_green_switch()
            return {'status': 'success', 'version': model_version}
        else:
            self._rollback_to_blue()
            raise DeploymentError("Performance degradation detected")
    
    def _gradual_traffic_shift(self) -> Dict[str, Any]:
        """Gradually shift traffic from blue to green."""
        traffic_percentages = [5, 10, 25, 50, 75, 100]
        
        for percentage in traffic_percentages:
            self._set_traffic_split(blue=100-percentage, green=percentage)
            time.sleep(300)  # Wait 5 minutes
            
            # Monitor key metrics
            metrics = self.monitoring.get_real_time_metrics()
            if self._detect_performance_degradation(metrics):
                self._set_traffic_split(blue=100, green=0)
                raise DeploymentError(f"Performance degradation at {percentage}% traffic")
        
        return {'status': 'completed', 'final_split': {'blue': 0, 'green': 100}}
```

## 6. 📊 **MONITORING & OBSERVABILITY WORKFLOW**

### 6.1 **Model Performance Monitoring**
```python
# monitoring/model_monitor.py
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ModelPerformanceMonitor:
    """Monitor ML model performance in production."""
    
    def __init__(self, model_name: str, monitoring_config: Dict):
        self.model_name = model_name
        self.config = monitoring_config
        self.alert_thresholds = monitoring_config['alert_thresholds']
        
    def monitor_real_time_performance(self) -> Dict[str, Any]:
        """Monitor model performance in real-time."""
        
        # Get recent predictions and actual outcomes
        recent_data = self._get_recent_performance_data()
        
        # Calculate performance metrics
        current_metrics = self._calculate_performance_metrics(recent_data)
        
        # Compare with baseline
        baseline_metrics = self._get_baseline_metrics()
        metric_drift = self._calculate_metric_drift(current_metrics, baseline_metrics)
        
        # Check for alerts
        alerts = self._check_performance_alerts(current_metrics, metric_drift)
        
        # Data drift detection
        data_drift_score = self._detect_data_drift(recent_data)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'metric_drift': metric_drift,
            'data_drift_score': data_drift_score,
            'alerts': alerts,
            'recommendations': self._generate_recommendations(alerts, metric_drift)
        }
    
    def _detect_data_drift(self, recent_data: pd.DataFrame) -> float:
        """Detect data drift using statistical tests."""
        reference_data = self._get_reference_data()
        
        drift_scores = []
        for column in recent_data.select_dtypes(include=[np.number]).columns:
            # KS test for numerical features
            ks_statistic, p_value = stats.ks_2samp(
                reference_data[column], recent_data[column]
            )
            drift_scores.append(ks_statistic)
        
        return np.mean(drift_scores)
    
    def _check_performance_alerts(self, 
                                 current_metrics: Dict, 
                                 metric_drift: Dict) -> List[Dict]:
        """Check for performance degradation alerts."""
        alerts = []
        
        for metric, value in current_metrics.items():
            threshold = self.alert_thresholds.get(metric)
            if threshold and value < threshold:
                alerts.append({
                    'type': 'performance_degradation',
                    'metric': metric,
                    'current_value': value,
                    'threshold': threshold,
                    'severity': 'high' if value < threshold * 0.9 else 'medium'
                })
        
        for metric, drift in metric_drift.items():
            if abs(drift) > 0.1:  # 10% drift threshold
                alerts.append({
                    'type': 'metric_drift',
                    'metric': metric,
                    'drift_percentage': drift * 100,
                    'severity': 'high' if abs(drift) > 0.2 else 'medium'
                })
        
        return alerts

# Monitoring configuration
monitoring_config = {
    'alert_thresholds': {
        'accuracy': 0.85,
        'f1_score': 0.80,
        'precision': 0.82,
        'recall': 0.78
    },
    'monitoring_frequency': '5m',
    'data_drift_threshold': 0.3,
    'alert_channels': ['slack', 'email', 'pagerduty']
}
```

### 6.2 **Automated Alerting System**
```markdown
🚨 **Alert Management Workflow:**

**Performance Alerts (Critical)**
- Model accuracy drops below 85%
- Prediction latency exceeds 500ms
- Error rate above 5%
- Data drift score > 0.3

**Business Alerts (High)**
- Prediction volume anomalies
- Revenue impact degradation
- User experience metrics decline
- SLA violations

**Technical Alerts (Medium)**
- Memory usage above 80%
- CPU utilization peaks
- Storage capacity warnings
- Network connectivity issues

**Alert Response Workflow:**
1. **Immediate:** Automated email/Slack notification
2. **5 minutes:** PagerDuty escalation if unacknowledged
3. **15 minutes:** Auto-rollback to previous model version
4. **30 minutes:** Executive team notification
5. **1 hour:** Incident response team activation
```

## 7. 🔄 **CONTINUOUS IMPROVEMENT WORKFLOW**

### 7.1 **Model Retraining Pipeline**
```python
# workflows/retraining_pipeline.py
class AutomatedRetrainingPipeline:
    """Automated model retraining with quality gates."""
    
    def __init__(self, model_config: Dict, schedule: str):
        self.model_config = model_config
        self.schedule = schedule  # e.g., "weekly", "monthly"
        self.performance_threshold = 0.85
        
    def should_retrain_model(self) -> Dict[str, bool]:
        """Determine if model should be retrained."""
        
        triggers = {
            'performance_degradation': self._check_performance_degradation(),
            'data_drift_detected': self._check_data_drift(),
            'scheduled_retrain': self._check_schedule_trigger(),
            'new_data_available': self._check_new_data_availability(),
            'business_requirement': self._check_business_triggers()
        }
        
        should_retrain = any(triggers.values())
        
        return {
            'should_retrain': should_retrain,
            'triggers': triggers,
            'recommendation': self._generate_retrain_recommendation(triggers)
        }
    
    def execute_retraining(self) -> Dict[str, Any]:
        """Execute automated retraining workflow."""
        
        retrain_decision = self.should_retrain_model()
        if not retrain_decision['should_retrain']:
            return {'status': 'skipped', 'reason': 'No retraining triggers met'}
        
        with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Step 1: Prepare updated training data
            training_data = self._prepare_training_data()
            
            # Step 2: Retrain model with new data
            new_model = self._retrain_model(training_data)
            
            # Step 3: Validate new model performance
            validation_results = self._validate_retrained_model(new_model)
            
            # Step 4: A/B test against current model
            ab_test_results = self._run_ab_test(new_model)
            
            # Step 5: Decide on deployment
            deployment_decision = self._make_deployment_decision(
                validation_results, ab_test_results
            )
            
            if deployment_decision['deploy']:
                # Step 6: Deploy new model
                deployment_result = self._deploy_new_model(new_model)
                return {
                    'status': 'success',
                    'validation_results': validation_results,
                    'ab_test_results': ab_test_results,
                    'deployment_result': deployment_result
                }
            else:
                return {
                    'status': 'rejected',
                    'reason': deployment_decision['reason'],
                    'validation_results': validation_results
                }
```

### 7.2 **Feedback Loop Integration**
```markdown
🔄 **Continuous Learning Workflow:**

**User Feedback Collection:**
- [ ] Implement explicit feedback mechanisms
- [ ] Track user interactions and behavior
- [ ] Collect domain expert annotations
- [ ] Monitor business outcome metrics

**Feedback Processing Pipeline:**
- [ ] Validate feedback quality and consistency
- [ ] Aggregate feedback across time periods
- [ ] Weight feedback by user expertise
- [ ] Identify patterns and trends

**Model Improvement Cycle:**
- [ ] Incorporate feedback into training data
- [ ] Adjust model objectives and metrics
- [ ] Retrain with updated datasets
- [ ] Validate improvement hypotheses

**Performance Tracking:**
- [ ] A/B test new model versions
- [ ] Monitor long-term performance trends
- [ ] Measure business impact improvements
- [ ] Document lessons learned
```

## 8. 📋 **WORKFLOW QUALITY GATES**

### 8.1 **Quality Gate Checklist**
```markdown
✅ **Data Quality Gates:**
- [ ] Data completeness > 90%
- [ ] Data accuracy validated against business rules
- [ ] No critical data quality issues
- [ ] Schema validation passed
- [ ] Data lineage documented

✅ **Model Quality Gates:**
- [ ] Performance metrics meet minimum thresholds
- [ ] Cross-validation results stable
- [ ] Bias evaluation completed and approved
- [ ] Model interpretability requirements met
- [ ] Security vulnerability scan passed

✅ **Deployment Quality Gates:**
- [ ] Integration tests passed (100%)
- [ ] Load testing completed successfully
- [ ] Security scanning approved
- [ ] Monitoring and alerting configured
- [ ] Rollback procedures tested

✅ **Production Quality Gates:**
- [ ] Health checks passing
- [ ] Performance metrics within SLA
- [ ] No critical alerts for 24 hours
- [ ] Business metrics showing positive impact
- [ ] Stakeholder approval obtained
```

### 8.2 **Automated Gate Validation**
```python
# workflows/quality_gates.py
class QualityGateValidator:
    """Automated validation of workflow quality gates."""
    
    def __init__(self, gate_config: Dict[str, Any]):
        self.gate_config = gate_config
        self.validation_results = {}
        
    def validate_all_gates(self, workflow_context: Dict) -> Dict[str, Any]:
        """Validate all quality gates for workflow stage."""
        
        gate_results = {}
        
        for gate_name, gate_config in self.gate_config.items():
            try:
                gate_result = self._validate_gate(gate_name, gate_config, workflow_context)
                gate_results[gate_name] = gate_result
            except Exception as e:
                gate_results[gate_name] = {
                    'passed': False,
                    'error': str(e),
                    'severity': 'critical'
                }
        
        overall_passed = all(result['passed'] for result in gate_results.values())
        
        return {
            'overall_passed': overall_passed,
            'gate_results': gate_results,
            'recommendations': self._generate_gate_recommendations(gate_results)
        }
    
    def _validate_gate(self, gate_name: str, gate_config: Dict, context: Dict) -> Dict:
        """Validate individual quality gate."""
        
        if gate_name == 'data_quality':
            return self._validate_data_quality_gate(gate_config, context)
        elif gate_name == 'model_performance':
            return self._validate_model_performance_gate(gate_config, context)
        elif gate_name == 'deployment_readiness':
            return self._validate_deployment_readiness_gate(gate_config, context)
        else:
            raise ValueError(f"Unknown gate type: {gate_name}")

# Example quality gate configuration
quality_gate_config = {
    'data_quality': {
        'completeness_threshold': 0.90,
        'accuracy_threshold': 0.95,
        'consistency_threshold': 0.98
    },
    'model_performance': {
        'min_accuracy': 0.85,
        'min_f1_score': 0.80,
        'max_bias_score': 0.1
    },
    'deployment_readiness': {
        'test_coverage_threshold': 0.90,
        'performance_test_passed': True,
        'security_scan_passed': True
    }
}
```

## 9. 🛠️ **WORKFLOW AUTOMATION TOOLS**

### 9.1 **Tool Integration Stack**
```yaml
# Workflow automation tool stack
automation_tools:
  orchestration:
    primary: "Apache Airflow"
    alternative: "Kubeflow Pipelines"
    cloud: "AWS Step Functions / Azure Logic Apps"
    
  version_control:
    code: "Git + GitHub/GitLab"
    data: "DVC (Data Version Control)"
    models: "MLflow Model Registry"
    
  ci_cd:
    pipeline: "GitHub Actions / GitLab CI"
    deployment: "ArgoCD / Flux"
    testing: "pytest + Great Expectations"
    
  monitoring:
    infrastructure: "Prometheus + Grafana"
    application: "Evidently AI + Seldon"
    business: "Custom dashboards + Tableau"
    
  collaboration:
    documentation: "Sphinx + GitBook"
    communication: "Slack + Microsoft Teams"
    project_management: "Jira + Linear"
```

### 9.2 **Workflow Automation Dashboard**
```python
# dashboard/workflow_dashboard.py
class WorkflowDashboard:
    """Central dashboard for workflow monitoring and control."""
    
    def __init__(self):
        self.workflow_status = {}
        self.performance_metrics = {}
        self.alert_manager = AlertManager()
        
    def get_workflow_overview(self) -> Dict[str, Any]:
        """Get comprehensive workflow status overview."""
        
        active_workflows = self._get_active_workflows()
        completed_workflows = self._get_completed_workflows_today()
        failed_workflows = self._get_failed_workflows()
        
        return {
            'summary': {
                'active_workflows': len(active_workflows),
                'completed_today': len(completed_workflows),
                'failed_today': len(failed_workflows),
                'success_rate': self._calculate_success_rate()
            },
            'active_workflows': active_workflows,
            'recent_failures': failed_workflows[-5:],  # Last 5 failures
            'performance_trends': self._get_performance_trends(),
            'resource_utilization': self._get_resource_utilization(),
            'upcoming_schedules': self._get_upcoming_schedules()
        }
    
    def trigger_workflow(self, workflow_name: str, parameters: Dict) -> Dict[str, Any]:
        """Manually trigger workflow execution."""
        
        # Validate workflow exists and parameters
        if not self._validate_workflow_trigger(workflow_name, parameters):
            raise ValueError("Invalid workflow or parameters")
        
        # Start workflow execution
        execution_id = self._start_workflow_execution(workflow_name, parameters)
        
        return {
            'execution_id': execution_id,
            'status': 'started',
            'estimated_duration': self._estimate_workflow_duration(workflow_name),
            'monitoring_url': f"/workflows/{execution_id}/monitor"
        }
```

## 10. 📊 **WORKFLOW METRICS & KPIs**

### 10.1 **Key Performance Indicators**
```markdown
📈 **Workflow Efficiency Metrics:**

**Lead Time Metrics:**
- Data-to-Model lead time: <7 days
- Model-to-Production lead time: <3 days
- Idea-to-Production lead time: <14 days
- Bug-to-Fix deployment time: <24 hours

**Quality Metrics:**
- Deployment success rate: >95%
- Rollback frequency: <5% of deployments
- Quality gate pass rate: >90%
- Production incident rate: <2 per month

**Performance Metrics:**
- Model accuracy in production: >85%
- Inference latency: <100ms P95
- System uptime: >99.9%
- Data pipeline SLA: >98%

**Business Metrics:**
- Time to business value: <30 days
- ROI on ML investments: >300%
- Stakeholder satisfaction: >8/10
- Model adoption rate: >80%
```

### 10.2 **Automated Reporting**
```python
# reporting/workflow_reporter.py
class WorkflowReporter:
    """Generate automated workflow performance reports."""
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate comprehensive weekly workflow report."""
        
        week_start = datetime.now() - timedelta(days=7)
        
        # Collect metrics
        workflow_metrics = self._collect_workflow_metrics(week_start)
        model_performance = self._collect_model_performance_metrics(week_start)
        business_impact = self._collect_business_impact_metrics(week_start)
        
        # Generate insights
        insights = self._generate_insights(workflow_metrics, model_performance)
        recommendations = self._generate_recommendations(insights)
        
        return {
            'report_period': f"{week_start.date()} to {datetime.now().date()}",
            'executive_summary': self._create_executive_summary(workflow_metrics),
            'workflow_performance': workflow_metrics,
            'model_performance': model_performance,
            'business_impact': business_impact,
            'insights': insights,
            'recommendations': recommendations,
            'action_items': self._generate_action_items(recommendations)
        }
    
    def _create_executive_summary(self, metrics: Dict) -> str:
        """Create executive summary of workflow performance."""
        
        summary = f"""
        ## Weekly AI/ML Workflow Summary
        
        ### Key Achievements:
        - Deployed {metrics['deployments_completed']} model updates
        - Achieved {metrics['average_accuracy']:.1%} average model accuracy
        - Maintained {metrics['uptime_percentage']:.1%} system uptime
        
        ### Performance Highlights:
        - {metrics['quality_gate_pass_rate']:.1%} quality gate pass rate
        - {metrics['avg_deployment_time']:.1f} hours average deployment time
        - {metrics['incident_count']} production incidents (target: <2)
        
        ### Areas for Improvement:
        - Focus on reducing deployment lead time
        - Improve automated testing coverage
        - Enhance monitoring and alerting capabilities
        """
        
        return summary.strip()
```

## 11. 🎯 **WORKFLOW BEST PRACTICES**

### 11.1 **DO's and DON'Ts**
```markdown
### **DO:**
✅ **Automate repetitive tasks** and manual processes
✅ **Implement comprehensive quality gates** at each stage
✅ **Track end-to-end workflow metrics** and KPIs
✅ **Use infrastructure as code** for reproducibility
✅ **Implement gradual rollouts** for model deployments
✅ **Monitor both technical and business metrics**
✅ **Document workflow decisions** and rationales
✅ **Test workflow components** in isolation
✅ **Implement proper error handling** and retries
✅ **Use semantic versioning** for models and pipelines

### **DON'T:**
❌ **Skip validation steps** to speed up deployment
❌ **Deploy models without proper testing**
❌ **Ignore data quality and drift monitoring**
❌ **Use manual processes** for critical workflows
❌ **Deploy to production** without staging validation
❌ **Forget to implement rollback procedures**
❌ **Ignore security and compliance requirements**
❌ **Mix experimental and production workflows**
❌ **Skip documentation** of workflow changes
❌ **Ignore feedback from previous workflow executions**
```

### 11.2 **Implementation Checklist**
```markdown
### **Week 1-2: Foundation Setup**
- [ ] Set up version control for code, data, and models
- [ ] Configure CI/CD pipeline with quality gates
- [ ] Implement basic monitoring and alerting
- [ ] Create workflow templates and standards
- [ ] Set up experiment tracking infrastructure

### **Week 3-4: Automation Implementation**
- [ ] Automate data validation and quality checks
- [ ] Implement automated model training pipelines
- [ ] Set up deployment automation with blue-green strategy
- [ ] Configure performance monitoring and alerting
- [ ] Create automated reporting dashboards

### **Week 5-6: Advanced Features**
- [ ] Implement A/B testing framework
- [ ] Set up automated retraining pipelines
- [ ] Configure advanced monitoring and drift detection
- [ ] Implement feedback loop integration
- [ ] Create comprehensive documentation

### **Week 7-8: Optimization & Scaling**
- [ ] Optimize workflow performance and resource usage
- [ ] Scale infrastructure for production workloads
- [ ] Implement advanced security and compliance features
- [ ] Train team on new workflow processes
- [ ] Conduct workflow performance review and optimization
```

