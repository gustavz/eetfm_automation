# eetfm_automation
eetfm_automation: Export and Evaluate TensorFlow Model Automation based on [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
> Note: Currently only faster_rcnn based models supported (e.g mask_rcnn)

## How to use this repository
- Place standard TF Object_detection API model directories inside the `base_models` folder, containing at least the following files:
    ```
    model.ckpt.data-00000-of-00001
    model.ckpt.index
    model.ckpt.meta
    pipeline.config
    ```
- Create a copy of `config.sample.yml` named `config.yml` and modify the params to your needs
- run `python eetfm_automation.py`
- the new models are exported to the `export_models` folder
- evaluation results are saved to a new `eval` folder inside each model
