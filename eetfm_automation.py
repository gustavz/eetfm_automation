"""
eetfm_automation: Export and Evaluate TensorFlow Model Automation
Based on TensorFlow Object Detection API

Written by github/GustavZ
"""
import os
import sys
import yaml
import datetime
import subprocess
import numpy as np
import tensorflow as tf

def get_models_list(models_dir):
    """
    Takes: Models Directory Path
    Returns: list of model names in models_path directory
    """
    for root, dirs, files in os.walk(models_dir):
        if root.count(os.sep) - models_dir.count(os.sep) == 0:
            for idx,model in enumerate(dirs):
                models_list=[]
                models_list.append(dirs)
                models_list = np.squeeze(models_list)
                models_list.sort()
    print("> Loaded following sequention of models: \n{models_list}".format(**locals()))
    return models_list

def create_config_override(fs_score,fs_iou,ss_score,ss_iou,proposals,num_examples,metrics):
    """
    Takes: first_stage_nms_score_threshold, first_stage_nms_iou_threshold,
            second_stage_nms_score_threshold, second_stage_nms_iou_threshold,
            num max proposals
    Returns: config override string
    """
    config_override = ' \
            model {{ \
                faster_rcnn {{ \
                    first_stage_nms_score_threshold: {fs_score} \
                    first_stage_nms_iou_threshold: {fs_iou} \
                    first_stage_max_proposals: {proposals} \
                    second_stage_post_processing {{ \
                        batch_non_max_suppression {{ \
                            score_threshold: {ss_score} \
                            iou_threshold: {ss_iou} \
                            max_detections_per_class: {proposals} \
                            max_total_detections: {proposals} \
                        }} \
                    }} \
                }} \
            }} \
            eval_config {{ \
              num_examples: {num_examples} \
              metrics_set: {metrics} \
              visualize_groundtruth_boxes: true \
            }}'.format(**locals())
    return config_override


def export_model(base_model_name,
                export_model_name,
                config_override,
                base_models_dir,
                export_models_dir,
                tf_dir):
    """
    Takes: base model name, export model name, model directory path, tensorflow object detection api path,
            config override string
    Exports new Model with TensorFlow object detection API
    """

    base_model_path = base_models_dir + "/" + base_model_name
    export_model_path = export_models_dir + "/" + export_model_name
    cmd = 'python {tf_dir}/export_inference_graph.py \
            --input_type=image_tensor \
            --pipeline_config_path={base_model_path}/pipeline.config \
            --trained_checkpoint_prefix={base_model_path}/model.ckpt \
            --output_directory={export_model_path} \
            --config_override={config_override}'.format(**locals())

    print ("> Exporting model {export_model_name}".format(**locals()))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print process.returncode
    if process.returncode == 0:
        print("> Export of model {export_model_name} complete".format(**locals()))
    else:
        print("> ERROR: Exportation of model {model_name} NOT complete".format(**locals()))


def evaluate_model(model_name,models_dir,tf_dir):
    """
    Takes: model name, model directory path, tensorflow object detection api path
    Evaluates Model with TensorFlow object detection API
    """

    model_path = models_dir + "/" + model_name
    cmd = 'python {tf_dir}/eval.py \
            --logtostderr \
            --pipeline_config_path={model_path}/pipeline.config \
            --checkpoint_dir={model_path}/ \
            --eval_dir={model_path}/eval \
            --run_once=True'.format(**locals())

    print ("> Evaluating model {model_name}".format(**locals()))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print process.returncode
    if process.returncode == 0:
        log_evaldate(model_path)
        print("> Evaluation of model {model_name} complete".format(**locals()))
    else:
        print("> ERROR: Evaluation of model {model_name} NOT complete".format(**locals()))

def log_evaldate(model_path):
    eval_date = datetime.datetime.now()
    file = open('{model_path}/eval_log.txt'.format(**locals()),mode='a')
    file.write("\nModel evaluated at: {eval_date}".format(**locals()))
    file.close()


def main():
    print("> Start eetfm_automation: Export and Evaluate TensorFlow Model Automation")
    # load global config params
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        with open("config.sample.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

    TF_ODAPI_DIR = cfg['TF_ODAPI_DIR']
    BASE_MODELS_DIR = cfg['BASE_MODELS_DIR']
    EXPORT_MODELS_DIR = cfg['EXPORT_MODELS_DIR']
    EXPORT_MODELS_LIST = cfg['EXPORT_MODELS_LIST']
    EVAL_MODELS_LIST = cfg['EVAL_MODELS_LIST']
    FS_SCORES_LIST = cfg['FS_SCORES_LIST']
    FS_IOUS_LIST = cfg['FS_IOUS_LIST']
    SS_SCORES_LIST = cfg['SS_SCORES_LIST']
    SS_IOUS_LIST = cfg['SS_IOUS_LIST']
    PROPOSALS_LIST = cfg['PROPOSALS_LIST']
    NUM_EXAMPLES = cfg['NUM_EXAMPLES']
    METRICS = cfg['METRICS']
    SKIP_EVALED = cfg['SKIP_EVALED']

    # load all models from base models directory if no specific list is given
    if not EXPORT_MODELS_LIST:
        EXPORT_MODELS_LIST = get_models_list(BASE_MODELS_DIR)

    ALREADY_EXPORTED_MODELS_LIST = get_models_list(EXPORT_MODELS_DIR)

    # export models if not set to False
    if not EXPORT_MODELS_LIST[0] is False:
        print("> Start Exporting models")
        # crazy exportation loop is crazy
        for base_model_name in EXPORT_MODELS_LIST:
            for proposals in PROPOSALS_LIST:
                for fs_score in FS_SCORES_LIST:
                    for fs_iou in FS_IOUS_LIST:
                        for ss_score in SS_SCORES_LIST:
                            for ss_iou in SS_IOUS_LIST:
                                suffix = "_{proposals}p_{fs_score}fs_{fs_iou}fiou_{ss_score}ss_{ss_iou}siou".format(**locals())
                                export_model_name = base_model_name +  suffix
                                if not export_model_name in ALREADY_EXPORTED_MODELS_LIST:
                                    config_override = create_config_override(fs_score,fs_iou,ss_score,ss_iou,proposals,NUM_EXAMPLES,METRICS)
                                    export_model(base_model_name,export_model_name,config_override,BASE_MODELS_DIR,EXPORT_MODELS_DIR,TF_ODAPI_DIR)
                                else:
                                    print("> Skipping. Model {export_model_name} already exported".format(**locals()))

    # load all models from export models directory if no specific list is given
    if not EVAL_MODELS_LIST:
        EVAL_MODELS_LIST = get_models_list(EXPORT_MODELS_DIR)

    # evaluate models if not set to False
    if not EVAL_MODELS_LIST[0] is False:
        print("> Start Evaluating models")
        for model_name in EVAL_MODELS_LIST:
            model_path = EXPORT_MODELS_DIR+"/"+model_name
            # check if model was already evaluated
            if os.path.exists(model_path+"/eval_log.txt") and SKIP_EVALED:
                print("> Skipping. Model {model_name} already evaluated".format(**locals()))
            else:
                evaluate_model(model_name,EXPORT_MODELS_DIR,TF_ODAPI_DIR)

    print("> eetfm_automation complete")


if __name__ == '__main__':
    main()
