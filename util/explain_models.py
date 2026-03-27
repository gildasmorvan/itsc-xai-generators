from pyxai import Learning, Explainer
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from IPython.display import display, HTML
import numpy as np


def display_reasons_for_model(learner, model, data, classes):

    for solution_label, solution_name in classes.items():
        instances = data
        explainer = Explainer.initialize(model)
        necessary = []
        relevant = []
        minimal = []
        for instance, prediction in instances:
            explainer.set_instance(instance)
            necessary_literals = explainer.necessary_literals()
            if necessary_literals not in necessary:
                necessary.append(necessary_literals)
            relevant_literals = explainer.relevant_literals()
            if relevant_literals not in relevant:
                relevant.append(relevant_literals)
            minimal_sufficient_reason = explainer.minimal_sufficient_reason(n=1)
            if minimal_sufficient_reason not in minimal:
                minimal.append(minimal_sufficient_reason)
        
        display(HTML(f"<h4>Explanations</4>"))
        for reason_type_names, reason_type_values in {"necessary":necessary, "relevant":relevant, "minimal":minimal}.items():
            display(HTML(f"<h5>{reason_type_names} reasons</h5>"))
            for reason in reason_type_values:
                if reason:
                    print('\u2022',*explainer.to_features(reason))