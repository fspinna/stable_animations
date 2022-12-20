from scipy.stats import norm
import numpy as np
import torch
from PIL import Image
import pathlib
import json
from tqdm.auto import tqdm
import os
import cv2
from IPython.display import HTML
from base64 import b64encode


def sample_neigh(x_mins, x_maxs, n=1000):
    normal_cdf = norm.cdf([x_mins, x_maxs]).T
    uniform_sample = np.vstack(
        [
            np.random.uniform(normal_cdf[i, 0], normal_cdf[i, 1], size=n)
            for i in range(normal_cdf.shape[0])
        ]
    ).T
    X = norm.ppf(uniform_sample)
    return X


def define_mins_maxs(X, radius):
    A = X - radius
    B = X + radius
    return A, B


def latents_to_pil(latents, vae):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def encode_text(prompts, tokenizer, text_encoder, device, maxlen=None):
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    inp = tokenizer(
        prompts,
        padding="max_length",
        max_length=maxlen,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(inp.input_ids.to(device))[0].half()


def prompt_to_latent(
    prompts,
    tokenizer,
    text_encoder,
    unet,
    scheduler,
    device,
    seed=100,
    steps=70,
    width=512,
    height=512,
):

    # Defining batch size
    bs = len(prompts)

    # Converting textual prompt to embedding
    text = encode_text(prompts, tokenizer, text_encoder, device)

    # Adding an unconditional prompt , helps in the generation process
    uncond = encode_text([""] * bs, tokenizer, text_encoder, device, text.shape[1])
    emb = torch.cat([uncond, text])

    # Setting the start_seed
    if seed:
        torch.manual_seed(seed)

    # Initiating random noise
    latents = torch.randn(
        (bs, unet.in_channels, width // 8, height // 8), dtype=torch.float16
    )

    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)

    # Adding noise to the latents
    latents = latents.to(device).half() * scheduler.init_noise_sigma

    return latents, emb


def denoise_step(latents, emb, ts, scheduler, unet, g=7.5):
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)

    # Predicting noise residual using U-Net
    with torch.no_grad():
        u, t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

    # Performing Guidance
    pred = u + g * (t - u)

    # Conditioning  the latents
    latents = scheduler.step(pred, ts, latents).prev_sample
    return latents


def generate_points(
    latents, device, interpolation_fn=None, n=0, radius=0.01, random_state=0
):
    if random_state is not None:
        np.random.seed(random_state)

    x_mins, x_maxs = define_mins_maxs(np.ravel(latents.cpu()), radius)
    X = sample_neigh(x_mins, x_maxs, 1)
    lat_reshaped = X.reshape(latents.shape)
    endpoint = torch.tensor(lat_reshaped, dtype=torch.float16).to(device)
    latents_list = [latents]
    weights = np.linspace(0, 1, n)

    if interpolation_fn is None:
        interpolation_fn = torch.lerp

    for i, weigth in enumerate(weights):
        if i != 0 and i != n - 1:
            latents_list.append(interpolation_fn(latents, endpoint, weigth))
    latents_list.append(endpoint)
    return latents_list


def denoise_steps(latents_list, emb, scheduler, unet, g=7.5):
    final_latents = list()
    for j, latent in enumerate(tqdm(latents_list)):
        for i, ts in enumerate(scheduler.timesteps):
            latent = denoise_step(latent, emb, ts, scheduler, unet, g=g)
        final_latents.append(latent)
    return final_latents


def prompt_to_images(
    prompt,
    tokenizer,
    text_encoder,
    unet,
    scheduler,
    vae,
    device,
    n=0,
    radius=0.01,
    g=7.5,
    start_seed=100,
    end_seed=100,
    steps=70,
    width=512,
    height=512,
    folder=None,
    interpolation_fn=None,
):
    if folder is not None:
        path = pathlib.Path(folder)
        pathlib.Path(path).mkdir(exist_ok=True, parents=True)
        params = {
            "prompt": prompt,
            "radius": radius,
            "g": g,
            "start_seed": start_seed,
            "n": n,
            "interpolation_fn": interpolation_fn,
            "end_seed": end_seed,
            "steps": steps,
            "width": width,
            "height": height,
        }
        with open(str(path / "params.json"), "w") as outfile:
            json.dump(params, outfile)
    latents, emb = prompt_to_latent(
        prompts=prompt,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
        device=device,
        seed=start_seed,
        steps=steps,
        width=width,
        height=height,
    )

    latents_list = generate_points(
        latents,
        device,
        interpolation_fn=interpolation_fn,
        n=n + 2,
        radius=radius,
        random_state=end_seed,
    )

    final_latents = denoise_steps(latents_list, emb, scheduler, unet, g=g)

    images = list()
    for i, latent in enumerate(final_latents):
        img = latents_to_pil(latent, vae)[0]
        if folder is not None:
            img.save(str(path / "frame_{:08d}.png".format(i)))
        images.append(img)
    return images


def imgs_to_video(imgs, video_name='video.mp4', fps=24):
    video_dims = (imgs[0].width, imgs[0].height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, fps, video_dims)
    for img in imgs:
        tmp_img = img.copy()
        video.write(cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR))
    video.release()


def display_video(file_path, width=512):
    compressed_vid_path = 'comp_' + file_path
    if os.path.exists(compressed_vid_path):
        os.remove(compressed_vid_path)
    os.system(f'ffmpeg -i {file_path} -vcodec libx264 {compressed_vid_path}')

    mp4 = open(compressed_vid_path, 'rb').read()
    data_url = 'data:simul2/mp4;base64,' + b64encode(mp4).decode()
    return HTML("""
    <video width={} controls>
          <source src="{}" type="video/mp4">
    </video>
    """.format(width, data_url))


