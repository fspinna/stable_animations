# Stable Animations
Generates interpolations between close instances in the latent space.

## Examples
Check out the [Jupyter Notebook](https://github.com/fspinna/stable_animations/blob/main/notebooks/example.ipynb)!
![Alt Text](https://github.com/fspinna/stable_animations/blob/main/notebooks/example.gif)

## How to install
Standard
```bash
pip install git+https://github.com/fspinna/stable_animations.git
```
With cuda (recommended)
```bash
pip install git+https://github.com/fspinna/stable_animations.git --extra-index-url https://download.pytorch.org/whl/cu116
```

## Sources
I took a lot of inspiration (and code) from these great articles:
- https://towardsdatascience.com/stable-diffusion-using-hugging-face-501d8dbdd8
- https://towardsdatascience.com/stable-diffusion-using-hugging-face-variations-of-stable-diffusion-56fd2ab7a265

## Thanks to
- Valentina for the latent sampling technique