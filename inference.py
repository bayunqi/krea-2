import os

import click
import torch
from safetensors.torch import load_file

from autoencoder import QwenAutoencoder
from encoder import Qwen3VLConditioner, TextEncoderConfig
from mmdit import SingleMMDiTConfig, SingleStreamDiT
from sampling import sample

single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

qwen3_vl_4b = TextEncoderConfig(model_id="Qwen/Qwen3-VL-4B-Instruct")
checkpoints = {
    "oss_raw": os.environ.get("OSS_RAW"),
    "oss_turbo": os.environ.get("OSS_TURBO"),
}


def _cuda_index(device: str) -> int | None:
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return None
    return 0 if parsed.index is None else parsed.index


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32

    cuda_index = _cuda_index(device)
    if cuda_index is None or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(cuda_index)
    return torch.bfloat16 if major >= 8 else torch.float16


def _default_text_device(device: str) -> str:
    cuda_index = _cuda_index(device)
    if cuda_index is None or not torch.cuda.is_available():
        return device
    if torch.cuda.device_count() <= 1:
        return device
    return "cuda:1" if cuda_index == 0 else "cuda:0"


def _parse_devices(devices: str | None, default: str) -> list[str]:
    if not devices:
        return [default]
    return [device.strip() for device in devices.split(",") if device.strip()]


def _place_mmdit(mmdit: SingleStreamDiT, devices: list[str], dtype: torch.dtype):
    if len(devices) == 1:
        return mmdit.to(device=devices[0], dtype=dtype).eval().requires_grad_(False)

    root_device = devices[0]
    mmdit.posemb.to(device=root_device, dtype=dtype)
    mmdit.first.to(device=root_device, dtype=dtype)
    mmdit.tmlp.to(device=root_device, dtype=dtype)
    mmdit.tproj.to(device=root_device, dtype=dtype)
    mmdit.txtfusion.to(device=root_device, dtype=dtype)
    mmdit.txtmlp.to(device=root_device, dtype=dtype)

    for i, block in enumerate(mmdit.blocks):
        block.to(device=devices[i * len(devices) // len(mmdit.blocks)], dtype=dtype)

    mmdit.last.to(device=devices[-1], dtype=dtype)
    return mmdit.eval().requires_grad_(False)


def _resolve_checkpoint(checkpoint: str) -> str:
    ckpt = checkpoints.get(checkpoint, checkpoint)
    if not ckpt:
        raise click.ClickException(
            f"Checkpoint '{checkpoint}' is not configured. Set OSS_RAW/OSS_TURBO "
            "or pass a checkpoint file path."
        )
    if not os.path.exists(ckpt):
        raise click.ClickException(f"Checkpoint does not exist: {ckpt}")
    return ckpt


def _pipeline(
    mmdit_config=single_mmdit_large_wide,
    text_encoder_config=qwen3_vl_4b,
    checkpoint="oss_raw",
    device="cuda:0",
    text_device=None,
    dit_devices=None,
    dtype=torch.float16,
):
    """Build the autoencoder, text encoder, and MMDiT, load weights, and move to GPU."""
    dit_devices = _parse_devices(dit_devices, device)
    text_device = text_device or _default_text_device(device)
    print(f"model device = {device}")
    print(f"DiT devices = {', '.join(dit_devices)}")
    print(f"text encoder device = {text_device}")
    print(f"dtype = {dtype}")

    ae = QwenAutoencoder()
    encoder = Qwen3VLConditioner(
        text_encoder_config.model_id,
        text_encoder_config.max_length,
        select_layers=text_encoder_config.select_layers,
    )

    # Build on meta, load to passed device
    with torch.device("meta"):
        mmdit = SingleStreamDiT(mmdit_config)

    ckpt = _resolve_checkpoint(checkpoint)
    print("ckpt =", ckpt)
    mmdit.load_state_dict(load_file(ckpt), strict=True, assign=True)
    mmdit = _place_mmdit(mmdit, dit_devices, dtype)
    ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)
    encoder = encoder.to(device=text_device, dtype=dtype).eval().requires_grad_(False)

    return mmdit, ae, encoder


@click.command(help="Generate images with Krea 2 (K2).")
@click.argument("prompt")
@click.option(
    "--steps", default=28, show_default=True, help="number of denoising steps"
)
@click.option(
    "--cfg",
    default=4.5,
    show_default=True,
    help="classifier-free guidance scale (0 disables CFG)",
)
@click.option(
    "--y1",
    default=0.5,
    show_default=True,
    help="timestep-shift mu at min resolution",
)
@click.option(
    "--y2",
    default=1.15,
    show_default=True,
    help="timestep-shift mu at max resolution",
)
@click.option("--width", default=1024, show_default=True)
@click.option("--height", default=1024, show_default=True)
@click.option(
    "--num-images",
    default=1,
    show_default=True,
    help="number of images to generate from the prompt",
)
@click.option(
    "--seed", default=0, show_default=True, help="base seed; image i uses seed + i"
)
@click.option(
    "--checkpoint",
    envvar="K2_CHECKPOINT",
    default="oss_raw",
    show_default=True,
    help="Checkpoint alias (oss_raw, oss_turbo) or a checkpoint file path",
)
@click.option(
    "--mu",
    default=None,
    help="timestep-shift mu",
    type=float,
)
@click.option(
    "--output", default="sample", show_default=True, help="output filename prefix"
)
@click.option(
    "--device",
    envvar="K2_DEVICE",
    default="cuda:0",
    show_default=True,
    help="device for the DiT sampler and VAE",
)
@click.option(
    "--text-device",
    envvar="K2_TEXT_DEVICE",
    default=None,
    help="device for the Qwen text encoder; defaults to another CUDA device when available",
)
@click.option(
    "--dit-devices",
    envvar="K2_DIT_DEVICES",
    default=None,
    help="comma-separated devices for DiT layer sharding, e.g. cuda:0,cuda:2,cuda:3",
)
@click.option(
    "--dtype",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "float16", "bfloat16", "float32"]),
    help="inference dtype; auto uses float16 on V100 and bfloat16 on Ampere+",
)
def main(
    prompt,
    steps,
    cfg,
    y1,
    y2,
    width,
    height,
    num_images,
    seed,
    checkpoint,
    output,
    mu,
    device,
    text_device,
    dit_devices,
    dtype,
):
    torch_dtype = _resolve_dtype(dtype, device)
    dit, ae, encoder = _pipeline(
        checkpoint=checkpoint,
        device=device,
        text_device=text_device,
        dit_devices=dit_devices,
        dtype=torch_dtype,
    )

    images = sample(
        dit,
        ae,
        encoder,
        [prompt] * num_images,
        width=width,
        height=height,
        steps=steps,
        guidance=cfg,
        seed=seed,
        y1=y1,
        y2=y2,
        mu=mu,
        device=device,
        dtype=torch_dtype,
    )
    for i, image in enumerate(images):
        out = f"{output}_{i}.png"
        image.save(out)
        click.echo(f"saved {out}")


if __name__ == "__main__":
    main()
