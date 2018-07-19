"""
eetfm_automation: Export and Evaluate TensorFlow Model Automation
Based on TensorFlow Object Detection API

Written by github/GustavZ
"""
import re
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
    models_list=[]
    for root, dirs, files in os.walk(models_dir):
        if root.count(os.sep) - models_dir.count(os.sep) == 0:
            for idx,model in enumerate(dirs):
                models_list=[]
                models_list.append(dirs)
    if len(models_list) == 1:
        models_list = models_list.pop(0)
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
    config_override = '"\
        model {{\n\
          faster_rcnn {{\n\
            first_stage_nms_score_threshold: {fs_score}\n\
            first_stage_nms_iou_threshold: {fs_iou}\n\
            first_stage_max_proposals: {proposals}\n\
            second_stage_post_processing {{\n\
              batch_non_max_suppression {{\n\
                score_threshold: {ss_score}\n\
                iou_threshold: {ss_iou}\n\
                max_detections_per_class: {proposals}\n\
                max_total_detections: {proposals}\n\
              }}\n\
            }}\n\
          }}\n\
        }}\n\
        eval_config {{\n\
          num_examples: {num_examples}\n\
          metrics_set: {metrics!r}\n\
          visualize_groundtruth_boxes: true\n\
        }}"'.format(**locals())
    return config_override


def subprocess_command(cmd,model_name,task,model_path=None):
    """
    starts command in a supprocess and does some extra stuff
    """
    print ("> Start {task} {model_name}".format(**locals()))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    #process.wait()
    if task in ['export','eval']:
        process.wait()
        print process.returncode
        if process.returncode == 0:
            if model_path is not None:
                log_date(model_path,task)

    elif task in ['summarize','benchmark']:
        proc_stdout = process.communicate()[0].strip()
        print proc_stdout
        process.wait()

    if process.returncode == 0:
        print("> {task} of model {model_name} complete".format(**locals()))
    else:
        print("> ERROR: {task} of model {model_name} NOT complete".format(**locals()))


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
    subprocess_command(cmd,export_model_name,'export')


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
    subprocess_command(cmd,model_name,'eval',model_path)


def log_date(model_path,task):
    """
    log evaluation date to txt file
    """
    eval_date = datetime.datetime.now()
    file = open('{model_path}/{task}_log.txt'.format(**locals()),mode='a')
    file.write("\nModel {task} at: {eval_date}".format(**locals()))
    file.close()


def benchmark_model(model_name,model_dir,tf_dir,shape):
    root_dir = os.getcwd()
    in_graph = '{root_dir}/{model_dir}/{model_name}/frozen_inference_graph.pb'.format(**locals())
    inputs = 'image_tensor'
    input_type = 'uint8'
    outputs = 'num_detections,detection_boxes,detection_scores,detection_classes,detection_masks'
    log_file = '{root_dir}/{model_dir}/{model_name}/model_benchmark.txt'.format(**locals())

    cmd = 'cd {tf_dir};\
    bazel run tensorflow/tools/benchmark:benchmark_model -- \
     --graph={in_graph} \
     --input_layer={inputs} \
     --input_layer_type={input_type} \
     --input_layer_shape={shape} \
     --output_layer={outputs} \
     --show_run_order=true \
     --show_time=true \
     --show_memory=true \
     --show_summary=true \
     --show_flops=true \
     2>&1 | tee {log_file}'. format(**locals())
    subprocess_command(cmd,model_name,'benchmark')


def summarize_model(model_name,model_dir,tf_dir):
    root_dir = os.getcwd()
    in_graph = '{root_dir}/{model_dir}/{model_name}/frozen_inference_graph.pb'.format(**locals())
    log_file = '{root_dir}/{model_dir}/{model_name}/model_summary.txt'.format(**locals())
    cmd ='cd {tf_dir};\
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
     --in_graph={in_graph} \
     2>&1 | tee {log_file}'.format(**locals())
    subprocess_command(cmd,model_name,'summarize')


def get_model_shape(model_path):
    """
    get input dimensions of model
    """
    f = open(model_path+'/pipeline.config')
    for i, line in enumerate(f):
        if i == 6:
            min = re.search(r'\d+', line).group()
        if i == 7:
            max = re.search(r'\d+', line).group()
            break
    f.close()
    shape = "1,{min},{max},3".format(**locals())
    return shape

def get_already_list(models_list,file_name,models_dir):
    already_models_list = []
    for model_name in models_list:
        if os.path.exists('{models_dir}/{model_name}/{file_name}'.format(**locals())):
            already_models_list.append(model_name)
    return already_models_list


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
    TF_DIR = cfg['TF_DIR']
    BENCHMARK_MODELS_LIST = cfg['BENCHMARK_MODELS_LIST']

    """
    MODEL EXPORTATION
    """
    # load all models from base models directory if no specific list is given
    if not EXPORT_MODELS_LIST:
        EXPORT_MODELS_LIST = get_models_list(BASE_MODELS_DIR)
    # export models if not set to False
    if not EXPORT_MODELS_LIST[0] is False:
        # count models already exported
        already_exported_models_list = get_models_list(EXPORT_MODELS_DIR)
        num_models_to_export = len(PROPOSALS_LIST)*len(FS_SCORES_LIST)*len(FS_IOUS_LIST)*len(SS_SCORES_LIST)*len(SS_IOUS_LIST)*len(EXPORT_MODELS_LIST)

        print("> Start Exporting {num_models_to_export} models".format(**locals()))
        # crazy exportation loop is crazy
        i = 0
        for base_model_name in EXPORT_MODELS_LIST:
            for proposals in PROPOSALS_LIST:
                for fs_score in FS_SCORES_LIST:
                    for fs_iou in FS_IOUS_LIST:
                        for ss_score in SS_SCORES_LIST:
                            for ss_iou in SS_IOUS_LIST:
                                suffix = "_{proposals}p_{fs_score}fs_{fs_iou}fiou_{ss_score}ss_{ss_iou}siou".format(**locals())
                                export_model_name = base_model_name +  suffix
                                # check if model is already exported
                                i += 1
                                print("> {}/{}".format(i,num_models_to_export))
                                if export_model_name in already_exported_models_list:
                                    print("> Skipping Export: {export_model_name} already exported".format(**locals()))
                                else:
                                    config_override = create_config_override(fs_score,fs_iou,ss_score,ss_iou,proposals,NUM_EXAMPLES,METRICS)
                                    export_model(base_model_name,export_model_name,config_override,BASE_MODELS_DIR,EXPORT_MODELS_DIR,TF_ODAPI_DIR)
    else:
        print("> Skipping Export: User Request")

    """
    MODEL EVALUATION
    """
    # load all models from export models directory if no specific list is given
    if not EVAL_MODELS_LIST:
        eval_all = True
        EVAL_MODELS_LIST = get_models_list(EXPORT_MODELS_DIR)
    # evaluate models if not set to False
    if not EVAL_MODELS_LIST[0] is False:

        # count models already evaluated
        already_evaled_models_list = get_already_list(EVAL_MODELS_LIST,'eval_log.txt',EXPORT_MODELS_DIR)
        if eval_all:
            num_models_to_eval = len(EVAL_MODELS_LIST) - len(already_evaled_models_list)
        else:
            num_models_to_eval = len(EVAL_MODELS_LIST)

        print("> Start Evaluating {num_models_to_eval} models".format(**locals()))
        for i,model_name in enumerate(EVAL_MODELS_LIST):
            # check if model is already evaluated
            print("> {}/{}".format(i+1,num_models_to_eval))
            if (model_name in already_evaled_models_list) and SKIP_EVALED:
                print("> Skipping Evaluation: {model_name} already evaluated".format(**locals()))
            else:
                evaluate_model(model_name,EXPORT_MODELS_DIR,TF_ODAPI_DIR)
    else:
        print("> Skipping Evaluation: User Request")

    """
    SUMMARIZE AND BENCHMARK MODEL
    """
    # load all models from export models directory if no specific list is given
    if not BENCHMARK_MODELS_LIST:
        benchmark_all = True
        BENCHMARK_MODELS_LIST = get_models_list(EXPORT_MODELS_DIR)
    if not BENCHMARK_MODELS_LIST[0] is False:
        # count models already evaluated
        already_benchmarked_models_list = get_already_list(BENCHMARK_MODELS_LIST,'model_benchmark.txt',EXPORT_MODELS_DIR)
        if benchmark_all:
            num_models_to_benchmark = len(BENCHMARK_MODELS_LIST) - len(already_benchmarked_models_list)
        else:
            num_models_to_benchmark = len(BENCHMARK_MODELS_LIST)

        print("> Start Summarizing and Benchmarking {num_models_to_benchmark} models".format(**locals()))
        for i,model_name in enumerate(BENCHMARK_MODELS_LIST):
            # check if model is already evaluated
            print("> {}/{}".format(i+1,num_models_to_benchmark))
            if model_name in already_benchmarked_models_list:
                print("> Skipping Benchmark test: {model_name} already benchmarked".format(**locals()))
            else:
                model_path = EXPORT_MODELS_DIR+"/"+model_name
                shape = get_model_shape(model_path)
                summarize_model(model_name,EXPORT_MODELS_DIR,TF_DIR)
                benchmark_model(model_name,EXPORT_MODELS_DIR,TF_DIR,shape)
    else:
        print("> Skipping Benchmarking: User Request")


    print("> eetfm_automation complete")


if __name__ == '__main__':
    main()
