is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (
        Variable,
        Function,
        using_config,
        no_grad,
        as_array,
        as_variable,
        setup_variable
    )

else:
    from dezero.core import (
        Variable,
        Parameter,
        Function,
        using_config,
        no_grad,
        as_array,
        as_variable,
        setup_variable,
    )
    from dezero.layers import (
        Layer
    )
    from dezero.models import (
        Model
    )
    import dezero.functions
    from dezero.dataloaders import (
        DataLoader
    )

    import dezero.core
    dezero.core.reshape_func = dezero.functions.reshape
    dezero.core.transpose_func = dezero.functions.transpose
    dezero.core.sum_func = dezero.functions.sum
    dezero.core.sum_to_func = dezero.functions.sum_to

    Variable.__getitem__ = dezero.functions.get_item


setup_variable()