import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import torchvision.transforms.functional as T
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from PIL import Image
from tqdm.auto import tqdm

from dataset import get_train_dataset, get_val_dataset
from model import Model
from utils import parse_args, setup_environment


def run_validation(
    val_dataloader,
    controlnet,
    pretrained_model_name,
    num_validation_images=1,
    num_samples=None,
    seed=None,
    weight_dtype=torch.float32,
):
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)

    image_logs = []
    num_eval_samples = num_samples or len(val_dataloader)
    pbar = tqdm(total=num_eval_samples)
    for batch in itertools.islice(val_dataloader, num_eval_samples):
        sketch, prompt = (batch["sketches"], batch["prompts"][0])

        images = []
        for _ in range(num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(prompt, sketch, num_inference_steps=20, generator=generator).images[0]
                images.append(image)

        image_logs.append(
            {
                "sketch": T.to_pil_image(sketch.squeeze(0)),
                "images": images,
                "prompt": prompt,
            }
        )

        pbar.update()

    return image_logs


def log_eval_result_to_tracker(image_logs, accelerator, step):
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images, prompt, sketch = log["images"], log["prompt"], log["sketch"]

                formatted_images = []
                formatted_images.append(np.asarray(sketch))
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(prompt, formatted_images, step, dataformats="NHWC")


def save_visual_results(image_logs, save_root):
    for idx, log in enumerate(image_logs):
        images = [log["sketch"], *log["images"]]
        width, height = images[0].size
        dst = Image.new("RGB", (width * len(images), height))
        for i, img in enumerate(images):
            dst.paste(img, (i * width, 0))
        dst.save(save_root.joinpath(f"{idx}.png"))


def main(args):
    logger, accelerator = setup_environment(args)
    logger.info(accelerator.state, main_process_only=False)

    val_dataloader = torch.utils.data.DataLoader(
        get_val_dataset(args.data_root, resolution=args.resolution),
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    model = Model(
        pretrained_model_name=args.pretrained_model_name,
        revision=args.revision,
        weight_dtype=weight_dtype,
    )
    model = model.to(accelerator.device)

    # evaluation
    if args.phase == "eval":
        assert args.controlnet_weight != None
        logger.info("Loading existing controlnet weights")
        model.load_controlnet_weight(args.controlnet_weight)

        image_logs = run_validation(
            val_dataloader,
            model.controlnet,
            pretrained_model_name=args.pretrained_model_name,
            seed=args.seed,
            num_validation_images=args.num_validation_images,
            num_samples=args.num_validation_samples,
            weight_dtype=weight_dtype,
        )
        save_root = Path(args.output_dir).joinpath("visuals")
        save_root.mkdir(parents=True, exist_ok=True)
        save_visual_results(image_logs, save_root)
        sys.exit(0)

    # training
    model.train()

    # NOTE: pytorch optimizer explicitly accepts parameter that requires grad
    # see https://github.com/pytorch/pytorch/issues/679
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = torch.utils.data.DataLoader(
        get_train_dataset(args.data_root, resolution=args.resolution),
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):

                # TODO: prompts can be preprocessed, or cached on the fly
                photos, sketches, prompts = (
                    batch["photos"].to(device=accelerator.device, dtype=weight_dtype),
                    batch["sketches"].to(device=accelerator.device, dtype=weight_dtype),
                    batch["prompts"],
                )
                loss = model(photos, sketches, prompts)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.controlnet.save_pretrained(args.output_dir)

                    if global_step % args.validation_steps == 0:
                        logger.info("Running validation... ")
                        # image_logs = run_validation(
                        #     val_dataloader,
                        #     controlnet=accelerator.unwrap_model(model).controlnet,
                        #     pretrained_model_name=args.pretrained_model_name,
                        #     seed=args.seed,
                        #     num_validation_images=args.num_validation_images,
                        #     num_samples=args.num_validation_samples,
                        #     weight_dtype=weight_dtype,
                        # )
                        # log_eval_result_to_tracker(image_logs, accelerator, global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.controlnet.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
