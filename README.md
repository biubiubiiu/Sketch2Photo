# Sketch2Photo

## Requirements

- PyTorch

```sh
pip install -r requirements.txt
```

## Known Issues

- throws CUDA error when using PyTorch 2.0

## Usage

```sh
# training
accelerate launch main_stable_diffusion.py --output_dir ${WORK_DIR}

# evaluation
accelerate launch main_stable_diffusion.py --phase eval --controlnet_weight ${WORK_DIR}
```