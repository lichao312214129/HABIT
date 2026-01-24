"""
Dependency Injection Container

A lightweight dependency injection container for managing service instances.
Supports factory functions, singleton pattern, and lazy initialization.
"""

from typing import Dict, Callable, Any, Optional


class DIContainer:
    """
    Lightweight dependency injection container.
    
    Manages service registration and instantiation with support for:
    - Factory functions for lazy initialization
    - Singleton pattern for shared instances
    - Explicit instance registration
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, bool] = {}
    
    def register(self, name: str, factory: Callable, singleton: bool = False):
        """
        Register a service factory function.
        
        Args:
            name: Service name for lookup
            factory: Factory function that creates the service instance
            singleton: If True, the service will be cached and reused
        """
        self._factories[name] = factory
        self._singletons[name] = singleton
    
    def register_instance(self, name: str, instance: Any):
        """
        Register an explicit service instance.
        
        Args:
            name: Service name for lookup
            instance: Pre-created service instance
        """
        self._services[name] = instance
        self._singletons[name] = True
    
    def get(self, name: str) -> Any:
        """
        Get a service instance.
        
        Args:
            name: Service name to retrieve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        if name in self._services and self._singletons.get(name, False):
            return self._services[name]
        
        if name not in self._factories:
            raise ValueError(f"Service '{name}' not registered in DI container")
        
        instance = self._factories[name]()
        
        if self._singletons.get(name, False):
            self._services[name] = instance
        
        return instance
    
    def has(self, name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            name: Service name to check
            
        Returns:
            True if service is registered, False otherwise
        """
        return name in self._factories or name in self._services
    
    def clear(self):
        """
        Clear all cached service instances.
        
        Useful for testing to reset state between tests.
        """
        self._services.clear()
    
    def list_services(self) -> list:
        """
        List all registered service names.
        
        Returns:
            List of service names
        """
        return list(set(self._factories.keys()) | set(self._services.keys()))
