"""
Data module for quantum circuit datasets
"""

# Base abstract classes
from .base_quantum_dataset import (
    BaseQuantumDataset,
    PropertyPredictionDataset,
    DecisionTransformerCompatDataset,
    AugmentedQuantumCompatDataset,
    create_property_dataset,
    create_decision_transformer_dataset,
    create_augmented_dataset
)

# Concrete implementations (backward compatibility)
from .quantum_circuit_dataset import (
    CircuitSpec,
    MeasurementResult, 
    CircuitData,
    QuantumCircuitDataset,
    DatasetManager,
    PropertyNormalizer,
    create_dataloaders
)

from .decision_transformer_dataset import DecisionTransformerDataset
from .simple_dt_collator import SimpleDecisionTransformerCollator
from .augmented_dataset import AugmentedQuantumDataset, create_augmented_datasets

__all__ = [
    # Abstract base classes
    'BaseQuantumDataset',
    'PropertyPredictionDataset', 
    'DecisionTransformerCompatDataset',
    'AugmentedQuantumCompatDataset',
    'create_property_dataset',
    'create_decision_transformer_dataset',
    'create_augmented_dataset',
    
    # Concrete implementations
    'CircuitSpec',
    'MeasurementResult',
    'CircuitData', 
    'QuantumCircuitDataset',
    'DatasetManager',
    'PropertyNormalizer',
    'create_dataloaders',
    'DecisionTransformerDataset',
    'SimpleDecisionTransformerCollator',
    'AugmentedQuantumDataset',
    'create_augmented_datasets'
]
