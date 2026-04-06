"""
DCNv2 - Deformable Convolutional Networks V2
"""

# Try to import the compiled DCN functions
try:
    from .functions import modulated_deform_conv, deform_conv
except ImportError:
    # Not compiled yet
    pass

# Try to import the DCN modules
try:
    from .modules.deform_conv import (
        DeformConv,
        DeformConvPack,
        ModulatedDeformConv,
        ModulatedDeformConvPack
    )
except ImportError:
    # Not compiled yet
    pass

__all__ = [
    'DeformConv',
    'DeformConvPack',
    'ModulatedDeformConv',
    'ModulatedDeformConvPack',
    'modulated_deform_conv',
    'deform_conv',
]
