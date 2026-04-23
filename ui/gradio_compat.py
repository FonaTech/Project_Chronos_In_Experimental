"""Optional Gradio compatibility layer.

The core smoke tests exercise helper functions from the UI modules in minimal
CI environments where the optional WebUI dependency may not be installed. In a
real WebUI install this module simply re-exports gradio. Without gradio it
provides a tiny no-op component surface so imports and build_app() smoke tests
still validate the wiring that does not require a browser runtime.
"""
from __future__ import annotations

from types import SimpleNamespace


try:  # pragma: no cover - exercised when gradio is installed
    import gradio as gr  # type: ignore
    HAS_GRADIO = True
except ModuleNotFoundError:  # pragma: no cover - exercised in slim CI
    HAS_GRADIO = False

    class _Component:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.value = kwargs.get("value")
            self.label = kwargs.get("label")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def click(self, *args, **kwargs):
            return self

        def change(self, *args, **kwargs):
            return self

        def tick(self, *args, **kwargs):
            return self

        def launch(self, **kwargs):
            self.launch_kwargs = kwargs
            return self

    def _update(**kwargs):
        return SimpleNamespace(**kwargs)

    gr = SimpleNamespace(
        Accordion=_Component,
        BarPlot=_Component,
        Blocks=_Component,
        Button=_Component,
        Checkbox=_Component,
        Column=_Component,
        Dropdown=_Component,
        File=_Component,
        HTML=_Component,
        JSON=_Component,
        Markdown=_Component,
        Number=_Component,
        Plot=_Component,
        Radio=_Component,
        Row=_Component,
        Slider=_Component,
        State=_Component,
        Tab=_Component,
        Textbox=_Component,
        Timer=_Component,
        update=_update,
    )

