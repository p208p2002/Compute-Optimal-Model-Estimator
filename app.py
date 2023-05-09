import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

description = """
Minimizing L under the constraint FLOPs(N, D) = C.

The functions $N_{opt}(C)$, and $D_{opt}(C)$ describe the optimal allocation of a computational budget $C$.

We use the following notation:

• L – the cross entropy loss in nats. Typically it will be averaged over the tokens in a context, but in
some cases we report the loss for specific tokens within the context.

• N – the number of model parameters, excluding all vocabulary and positional embeddings

• D – the dataset size in tokens

• C ≈ 6ND – an estimate of the total non-embedding training compute

$$E=1.69, A=406.4, \\alpha=0.34, \\beta=0.28$$
$$C\\approx6DN$$
$$L(N,D)=E+\\frac{A}{N^\\alpha}+\\frac{B}{D^\\beta}$$
$$N_{opt}(C),D_{opt}(C)={\\arg\\min}_{N,D\ s.t.\ FLOP/s(N,D)=C}\ L(N,D)$$

"""

article = """
References
- [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)
- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)
"""


def L(N, D):
    """ 
    Approximates loss given N parameters and D dataset size (in tokens),
    per Chinchilla paper.
    """
    E = 1.69  # entropy of natural language, limit of infinite model on infinite data
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    return A / (N ** alpha) + B / (D ** beta) + E


def plot_pens(tflpos_card, utilization, num_gps, training_days):
    fig = plt.figure()
    tflpos_card = float(tflpos_card)*(10**12)
    utilization = float(utilization)
    num_gps = int(num_gps)
    training_days = float(training_days)

    # target compute budget (usually know this because we know how many GPU for how long go brrr)
    c = tflpos_card*num_gps*86400*training_days*utilization

    # (I got this flop number from row 1 of Table A3)
    # sweep model sizes from 10M to 100B
    ns = 10 ** np.arange(7, 11, step=2**-4)
    # using C = 6*N*D, solve for D that maintains the compute budget c
    ds = c / (6 * ns)
    # evaluate the loss in each case
    losses = L(ns, ds)
    # find the argmin
    best = np.argmin(losses)

    best_model_size = f"{ns[best]/1e6:.2f}M"
    best_dataset_size = f"{ds[best]/1e9:.2f}B"

    # plot the loss
    # plt.figure(figsize=(3,3))
    plt.plot(ns, losses)
    plt.xscale('log')
    # plot a vertical bar at the best model size
    plt.axvline(ns[best], color='red')
    plt.xlabel('model size')
    plt.ylabel('loss')

    return fig, c, round(losses[best], 3), best_model_size, best_dataset_size


if __name__ == "__main__":
    iface = gr.Interface(
        fn=plot_pens,
        inputs=[
            gr.Textbox(label="TFLOP/s pre Card",value="40"),
            gr.Slider(label="System Utilization", minimum=0, maximum=1, step=0.01,value=0.25),
            gr.Textbox(label="Number of cards",value="1"),
            gr.Textbox(label="Training Days",value="7")
        ],
        outputs=[
            gr.Plot(label="Estimated Loss"),
            gr.Label(label="Total Compute Budget"),
            gr.Label(label="Estimated Final Loss"),
            gr.Label(label="Optimal Model Size"),
            gr.Label(label="Optimal Dataset Size (Tokens)")
        ],
        title="Compute-Optimal Model Estimator",
        description=description,
        article=article,
        theme='peach',
        live=False
    ).launch()
