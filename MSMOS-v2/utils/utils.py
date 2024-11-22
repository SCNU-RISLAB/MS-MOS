#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import math
import yaml
import shutil
import argparse

from decimal import Decimal
from datetime import datetime
from pip._vendor.distlib.compat import raw_input


def remove_exponent(d):
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    millnames = ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y']
    if prefixes:
        millnames = ['']
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    result = '{:.{precision}f}'.format(n / 10**(3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return '{0}{dx}'.format(result, dx=millnames[millidx])


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')


def load_yaml(path):
    try:
        print(f"\033[32m Opening arch config file {path}\033[0m")
        yaml_data = yaml.safe_load(open(path, 'r'))
        return yaml_data
    except Exception as e:
        print(e)
        print(f"Error opening {path} yaml file.")
        quit()


def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def check_pretrained_dir(path):
    if path is not None:
        if os.path.isdir(path):
            print("\033[32m model folder exists! Using model from %s \033[0m" % (path))
        else:
            print("\033[32m model folder doesnt exist! Start with random weights...\033[0m")
    else:
        print("\033[32m No pretrained directory found.\033[0m")


def check_model_dir(path):
    if path is not None:
        if os.path.isdir(path):
            print("\033[32m model folder exists! Using model from %s \033[0m" % (path))
        else:
            print("\033[32m model folder doesnt exist! Can't infer...\033[0m")
            quit()
    else:
        print("\033[32m No model directory found.\033[0m")


def make_logdir(FLAGS, resume_train=False):
    try:
        if resume_train:
            if FLAGS.pretrained == "":
                FLAGS.pretrained = None
                if os.path.isdir(FLAGS.log):
                    if os.listdir(FLAGS.log):
                        answer = raw_input("Log Directory is not empty. Do you want to proceed? [y/n]  ")
                        if answer == 'n':
                            quit()
                        else:
                            shutil.rmtree(FLAGS.log)
                os.makedirs(FLAGS.log)
            else:
                FLAGS.log = FLAGS.pretrained
                print("Not creating new log file. Using pretrained directory")
        else:
            if os.path.isdir(FLAGS.log):
                if os.listdir(FLAGS.log):
                    answer = raw_input("Log Directory is not empty. Do you want to proceed? [y/n]  ")
                    if answer == 'n':
                        quit()
                    else:
                        shutil.rmtree(FLAGS.log)
            os.makedirs(FLAGS.log)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()


def backup_to_logdir(FLAGS, pretrain_model=False):
    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("Copying files to %s for further reference." % FLAGS.log)
        shutil.copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
        shutil.copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")

        # Backup training code for later review
        code_backup_path = f"{FLAGS.log}/code"
        check_and_makedirs(code_backup_path)
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '*.py'))} {code_backup_path}")
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'common'))} {code_backup_path}")
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'config'))} {code_backup_path}")
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'modules'))} {code_backup_path}")
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'train_yaml'))} {code_backup_path}")
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'utils'))} {code_backup_path}")
        os.system(f"cp -r {os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'script'))} {code_backup_path}")
        if pretrain_model:
            shutil.copyfile(FLAGS.pretrained + "/MotionRV_valid_best", FLAGS.log + "/MotionRV_valid_best")

    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()


def make_predictions_dir(FLAGS, DATA, rm_old=False):
    # create predictions file folder
    try:
        if rm_old:
            if os.path.isdir(FLAGS.log):
                shutil.rmtree(FLAGS.log)
            os.makedirs(FLAGS.log)
            os.makedirs(os.path.join(FLAGS.log, "sequences"))
        check_and_makedirs(os.path.join(FLAGS.log, "sequences"))
        
        for seq in DATA["split"][FLAGS.split]:
            seq = '{0:02d}'.format(int(seq))
            print(f"{FLAGS.split} : {seq}")
            check_and_makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))

    except Exception as e:
        print(e)
        print("Error creating predictions directory. Check permissions!")
        raise

    pass


def get_args(flags=None):
    splits = ["train", "valid", "test"]

    if flags == "train":
        parser = argparse.ArgumentParser("./train.py")
    elif flags == "infer":
        parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--dataset', '-d', type=str,
        required=False,
        default='/data/datasets/kitti/',
        help='Dataset to train with. The parent directory of sequences. No Default.')
    
    parser.add_argument(
        '--log', '-l', type=str,
        default="./log_default" + datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the log data. Default: ./log_default/date+time')
    
    parser.add_argument(
        '--name', '-n', type=str,
        default="",
        help='If you want to give an aditional discriptive name')
    # parser.add_argument(
    #     '--uncertainty', '-u',
    #     required=False,
    #     type=str2bool, nargs='?',
    #     const=True, default=True,
    #     help='Set this if you want to use the Uncertainty Version')
    
    if flags == "train":
        parser.add_argument(
            '--pretrained', '-p', type=str,
            required=False,
            default='/data/zlf/mos3d/MotionRV-1stage/pretrain/',
            help='Directory to get the pretrained model. If not passed, do from scratch!')
        parser.add_argument(
            '--arch_cfg', '-ac', type=str,
            required=False,
            default='train_yaml/mos_pointrefine_stage.yml',
            help='Architecture yaml cfg file. See /config/arch for sample. No default.')
        
        parser.add_argument(
            '--data_cfg', '-dc', type=str,
            required=False,
            default='config/labels/semantic-kitti-mos.raw.yaml',
            help='Classification yaml cfg file. See /config/labels for sample. No default.')

    if flags == "infer":
        # parser.add_argument(
        #     '--monte-carlo', '-c',
        #     type=int, default=30,
        #     help='Number of samplings per scan.')
        parser.add_argument(
            '--model', '-m',
            type=str,
            required=True,
            default=None,
            help='Directory to get the trained model.')
        parser.add_argument(
            '--split', '-s',
            type=str,
            required=False,
            default=None,
            help='Split to evaluate on. One of ' +
            str(splits) + '. Defaults to %(default)s',)
        parser.add_argument(
            '--pointrefine', '-prf',
            action='store_true',
            required=False,
            help='Whether to use the PointHead module to refine predictions')

    return parser

def get_args_for_fusion(flags=None):
    splits = ["train", "valid", "test"]

    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--dataset', '-d', type=str,
        default='/data3/zlf_data/data_odometry_velodyne/dataset/',
        help='Dataset to train with. The parent directory of sequences. No Default.')
    parser.add_argument(
        '--log', '-l', type=str,
        default="./prediction_hqb_with_intensity" + datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the log data. Default: ./log_default/date+time')
    parser.add_argument(
        '--name', '-n', type=str,
        default="",
        help='If you want to give an aditional discriptive name')
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        default='/media/arc/Project/pycharm_project/SwinMotionSeg/predmodel',
        help='Directory to get the trained model.')
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default='valid',
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s', )
    parser.add_argument(
        '--pointrefine', '-prf',
        action='store_true',
        required=False,
        default=False,
        help='Whether to use the PointHead module to refine predictions')
    parser.add_argument(
        '--radius', '-r',
        type=int,
        required=False,
        default=-1,
        help='Calculated range radius, -1 means using all points.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )
    parser.add_argument(
        '--codalab',
        dest='codalab',
        type=str,
        default=None,
        help='Exports "scores.txt" to given output directory for codalab'
             'Defaults to %(default)s',
    )
    return parser