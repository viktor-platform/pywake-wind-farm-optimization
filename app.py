from viktor import ViktorController
from viktor.parametrization import ViktorParametrization


class Parametrization(ViktorParametrization):
    pass


class Controller(ViktorController):
    label = "wind farm"
    parametrization = Parametrization
