from .cnn_baseline import CNNBaseline

try:
    from .vim import VisionMambaSeg
    __all__ = ['VisionMambaSeg', 'CNNBaseline']
except ImportError:
    # Mamba compilation not available, CNN baseline still works
    __all__ = ['CNNBaseline']