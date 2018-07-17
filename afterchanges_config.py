"""
Scirpts to make changes to all model config files in a directory

"""
import os
import sys
import yaml
import fileinput


def main():
    # load global config params
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        with open("config.sample.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

    CHANGES_DIR = cfg['CHANGES_DIR']
    CHANGE_STRING_BEFORE = cfg['CHANGE_STRING_BEFORE']
    CHANGE_STRING_AFTER = cfg['CHANGE_STRING_AFTER']

    for root, dirs, files in os.walk(CHANGES_DIR):
        for file in files:
            if file.endswith(".config"):
                config_path = root+"/"+file
                print config_path
                for line in fileinput.input(config_path, inplace=1):
                    sys.stdout.write(line.replace(CHANGE_STRING_BEFORE,CHANGE_STRING_AFTER))


if __name__ == '__main__':
    main()
