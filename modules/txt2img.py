import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, height: int, width: int, *args):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[prompt_style, prompt_style2],
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
    )

    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"Prompt Style: {prompt_style}")
    print(f"Prompt Style 2: {prompt_style2}")
    print(f"Steps: {steps}")
    print(f"Sampler Index: {sampler_index}")
    print(f"Restore Faces: {restore_faces}")
    print(f"Tiling: {tiling}")
    print(f"n_iter: {n_iter}")
    print(f"batch_size: {batch_size}")
    print(f"cfg_scale: {cfg_scale}")
    print(f"seed: {seed}")
    print(f"subseed: {subseed}")
    print(f"subseed_strength: {subseed_strength}")
    print(f"seed_resize_from_h: {seed_resize_from_h}")
    print(f"seed_resize_from_w: {seed_resize_from_w}")
    print(f"height: {height}")
    print(f"width: {width}")

    # Print function args
    print("args:")
    for arg in args:
        print(f"'{arg}'")


    print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is not None:
        pass
    else:
        processed = process_images(p)

    shared.total_tqdm.clear()

    return processed.images, processed.js(), plaintext_to_html(processed.info)

