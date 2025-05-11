from typing import Dict, Type, Any, List
from .base_preprocessor import BasePreprocessor

class PreprocessorFactory:
    """Factory class for creating preprocessors.
    
    This class manages the registration and instantiation of preprocessors.
    """
    
    _preprocessors: Dict[str, Type[BasePreprocessor]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """Register a preprocessor class with the factory.
        
        Args:
            name (str): Name of the preprocessor to register.
            
        Returns:
            callable: Decorator function.
        """
        def decorator(preprocessor_class: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
            cls._preprocessors[name] = preprocessor_class
            return preprocessor_class
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BasePreprocessor:
        """Create an instance of a registered preprocessor.
        
        Args:
            name (str): Name of the preprocessor to create.
            **kwargs: Additional arguments to pass to the preprocessor constructor.
            
        Returns:
            BasePreprocessor: Instance of the requested preprocessor.
            
        Raises:
            ValueError: If the requested preprocessor is not registered.
        """
        if name not in cls._preprocessors:
            raise ValueError(f"Preprocessor '{name}' is not registered")
        return cls._preprocessors[name](**kwargs)
    
    @classmethod
    def get_available_preprocessors(cls) -> List[str]:
        """Get a list of all registered preprocessor names.
        
        Returns:
            List[str]: List of registered preprocessor names.
        """
        return list(cls._preprocessors.keys()) 