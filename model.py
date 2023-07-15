import random

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig


def import_model_class_from_model_name_or_path(pretrained_model_name: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompts(tokenizer, prompts, proportion_empty_prompts):
    prompts_to_tokenize = []
    for prompt in prompts:
        if random.random() < proportion_empty_prompts:
            prompts_to_tokenize.append("")
        elif isinstance(prompt, str):
            prompts_to_tokenize.append(prompt)
        else:
            raise ValueError(f"Prompts `{prompts}` should contain either strings or lists of strings.")
    inputs = tokenizer(
        prompts_to_tokenize,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


class Model(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        revision=None,
        prompt=None,
        proportion_empty_prompts=0,
        prompt_tuning=False,
        weight_dtype=torch.float32,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
        text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name, revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_name, subfolder="text_encoder", revision=revision
        )
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae", revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet", revision=revision)
        self.controlnet = ControlNetModel.from_unet(self.unet)

        # freeze parameters
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        self.prmopt = prompt
        self.proportion_empty_prompts = proportion_empty_prompts
        self.prompt_tuning = prompt_tuning

        # mixed precision
        self.weight_dtype = weight_dtype
        self.vae.to(dtype=weight_dtype)
        self.unet.to(dtype=weight_dtype)
        self.text_encoder.to(dtype=weight_dtype)

        if self.prompt_tuning:
            assert self.prmopt is not None
            self.prompt_embedding = self.get_prompt_embedding(self.prmopt)
            self.prompt_embedding = nn.Parameter(self.prompt_embedding)

    def load_controlnet_weight(self, controlnet_weight):
        self.controlnet = ControlNetModel.from_pretrained(controlnet_weight)

    def prepare_latent_diffusion(self, photos):
        # Convert images to latent space
        latents = self.vae.encode(photos).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents, latents, noise, timesteps

    def get_prompt_embedding(self, prompt):
        tokenized_prompts = tokenize_prompts(self.tokenizer, prompt, self.proportion_empty_prompts)
        text_embedding = self.text_encoder(tokenized_prompts.to(self.text_encoder.device))[0]
        return text_embedding

    def forward(self, photos, sketches, prompts):
        noisy_latents, latents, noise, timesteps = self.prepare_latent_diffusion(photos)
        encoder_hidden_states = self.get_prompt_embedding(prompts)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=sketches,
            return_dict=False,
        )

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
        ).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
