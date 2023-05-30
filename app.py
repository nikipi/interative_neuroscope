from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

import streamlit as st

model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(model_name)


def get_neuron_acts(text, layer, neuron_index):
    # Hacky way to get out state from a single hook - we have a single element list and edit that list within the hook.
        cache = {}

        def caching_hook(act, hook):
            cache["activation"] = act[0, :, neuron_index]

        model.run_with_hooks(
            text, fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
        )
        return to_numpy(cache["activation"])

def get_layers_acts(text, layer):
    # Hacky way to get out state from a single hook - we have a single element list and edit that list within the hook.
        cache = {}

        def caching_hook(act, hook):
            cache["activation"] = act[0, :, : ]

        model.run_with_hooks(
            text, fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
        )
        return to_numpy(cache["activation"])



# Create a Streamlit app
def main():
    

    st.set_option('deprecation.showPyplotGlobalUse', False)

# This is some CSS (tells us what style) to give each token a thin gray border,
# to make it easy to see token separation
    style_string = """<style> 
    span.token {
        border: 1px solid rgb(123, 123, 123)
        } 
    </style>"""



    def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val],
    # normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red
    # (0 = white, 1 = red). We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
        normalized_val = (val - min_val) / max_val
        return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"


    def basic_neuron_vis(text, layer, neuron_index, max_val=None, min_val=None):
        """
        text: The text to visualize
        layer: The layer index
        neuron_index: The neuron index
        max_val: The top end of our activation range, defaults to the maximum activation
        min_val: The top end of our activation range, defaults to the minimum activation

        Returns a string of HTML that displays the text with each token colored according to its activation

        Note: It's useful to be able to input a fixed max_val and min_val because otherwise the colors will change as you edit the text, which is annoying.
        """
        if layer is None:
            return "Please select a Layer"
        if neuron_index is None:
            return "Please select a Neuron"
        
        acts = get_neuron_acts(text, layer, neuron_index)

        act_max = acts.max()
        act_min = acts.min()

        # Defaults to the max and min of the activations
        if max_val is None:
            max_val = act_max
        if min_val is None:
            min_val = act_min
        # We want to make a list of HTML strings to concatenate into our final HTML string
        # We first add the style to make each token element have a nice border
        htmls = [style_string]
        # We then add some text to tell us what layer and neuron we're looking at - we're just dealing with strings and can use f-strings as normal
        # h4 means "small heading"
        htmls.append(f"<h4>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron_index}</b></h4>")
        # We then add a line telling us the limits of our range
        htmls.append(
            f"<h4>Max Range: <b>{max_val:.4f}</b>. Min Range: <b>{min_val:.4f}</b></h4>"
        )
        # If we added a custom range, print a line telling us the range of our activations too.
        if act_max != max_val or act_min != min_val:
            htmls.append(
                f"<h4>Custom Range Set. Max Act: <b>{act_max:.4f}</b>. Min Act: <b>{act_min:.4f}</b></h4>"
            )
        # Convert the text to a list of tokens
        str_tokens = model.to_str_tokens(text)

        for tok, act in zip(str_tokens, acts):
            # A span is an HTML element that
                # lets us style a part of a string (and remains on the same line by default)
            # We set the background color of the span to be the color we calculated from the activation
            # We set the contents of the span to be the token
            htmls.append(
                f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val)}' >{tok}</span>"
            )

        return "".join(htmls)

    st.title("Neuron Visualization")

    # Text input
    text = st.text_area("Enter text:", value="")
    str_tokens = model.to_str_tokens(text)

    # Layer selection
    layer = st.slider(
    "Layer index:",
    0, 9)

    # Neuron selection
    neuron_index = st.number_input("Neuron index:", value=0, min_value=0)


    neuron_html = basic_neuron_vis(text, layer, neuron_index)


    def create_heatmap2(text, layer, neuron_index, max_val=None, min_val=None):
        acts = get_layers_acts(text, layer)
        act_max = acts.max()
        act_min = acts.min()
        fig = go.Figure(data=go.Heatmap(z=acts.T, colorscale='RdBu', zmin=act_min, zmax=act_max))

        # set axis labels and title
        fig.update_layout(
            title='Activations Heatmap',
            xaxis_title='Tokens',
            yaxis_title='Neuron',
            width=800,
            height=1000,
        )

        fig.update_xaxes(tickangle=90,
                tickvals=np.arange(0, len(str_tokens)),
                ticktext=str_tokens)

        # set color bar range based on the range of values in the dataframe
        fig.update_traces(colorbar=dict(
            tickvals=[act_min, (act_min+act_max)/2, act_max],
            ticktext=[f"{act_min:.2f}", f"{(act_min+act_max)/2:.2f}", f"{act_max:.2f}"],
            tickmode='array',
            ticks='outside',
            thickness=20,
            len=0.5,
            title='Activation'
        ),hovertemplate='Layer: %{y}<br>Neuron: %{x}<br>Activation: %{z:.2f}')

        

        st.plotly_chart(fig)



    # Return the heatmap plot object

    st.markdown(neuron_html, unsafe_allow_html=True)
    create_heatmap2(text, layer, neuron_index, max_val=None, min_val=None)


    
if __name__ == "__main__":
    main()

