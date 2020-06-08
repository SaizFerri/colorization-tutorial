#!/usr/bin/env python3

import json

SSH_SERVER = 'dt1.f4.htw-berlin.de'
SSH_AUTH = {'username': '{{ssh_username}}', 'password': '{{ssh_password}}'}
DATA_DIR = 'colorization-tutorial'
AGENCY_URL = 'https://agency.f4.htw-berlin.de/dt'
# LEARNING_RATES = [0.0001, 0.0005]
# STEPS_PER_EPOCH = 10


batches = []

# for i, learning_rate in enumerate(LEARNING_RATES):
batch = {
    'inputs': {
        'data_dir': {
            'class': 'Directory',
            'connector': {
                'command': 'red-connector-ssh',
                'mount': True,
                'access': {
                    'host': SSH_SERVER,
                    'auth': SSH_AUTH,
                    'dirPath': DATA_DIR
                }
            }
        },
        'outputs_dir': {
            'class': 'Directory',
            'connector': {
                'command': 'red-connector-ssh',
                'mount': True,
                'access': {
                  'host': SSH_SERVER,
                  'auth': SSH_AUTH,
                  'dirPath': 'colorization-tutorial/outputs',
                  'writable': True
                }
            }
        },
        'checkpoints_dir': {
            'class': 'Directory',
            'connector': {
                'command': 'red-connector-ssh',
                'mount': True,
                'access': {
                  'host': SSH_SERVER,
                  'auth': SSH_AUTH,
                  'dirPath': 'colorization-tutorial/checkpoints',
                  'writable': True
                }
            }
        },
        'batch_size': 64,
        # 'learning_rate': learning_rate,
        # 'steps_per_epoch': STEPS_PER_EPOCH,
        'log_dir': {
            'class': 'Directory',
            'connector': {
                'command': 'red-connector-ssh',
                'mount': True,
                'access': {
                  'host': SSH_SERVER,
                  'auth': SSH_AUTH,
                  'dirPath': 'colorization-tutorial/log',
                  'writable': True
                }
            }
        },
        'log_file_name': 'training_0.log'
    },
    'outputs': {
        'weights_file': {
            'class': 'File',
            'connector': {
                'command': 'red-connector-ssh',
                'access': {
                    'host': SSH_SERVER,
                    'auth': SSH_AUTH,
                    'filePath': 'checkpoints/model.pth',
                }
            }
        },
        'gray_image_file': {
            'class': 'File',
            'connector': {
                'command': 'red-connector-ssh',
                'access': {
                    'host': SSH_SERVER,
                    'auth': SSH_AUTH,
                    'filePath': 'outputs/gray/img.jpg',
                }
            }
        },
        'color_image_file': {
            'class': 'File',
            'connector': {
                'command': 'red-connector-ssh',
                'access': {
                    'host': SSH_SERVER,
                    'auth': SSH_AUTH,
                    'filePath': 'outputs/color/img.jpg',
                }
            }
        }
    }
}
batches.append(batch)

with open('image_colorization.cwl.json') as f:
    cli = json.load(f)

red = {
    'redVersion': '9',
    'cli': cli,
    'batches': batches,
    'container': {
        'engine': 'docker',
        'settings': {
            'image': {
                'url': 'saizferri/colorization-tutorial',
            },
            'ram': 60000,
            'gpus': {
                'vendor': 'nvidia',
                'count': 1
            }
        }
    },
    'execution': {
        'engine': 'ccagency',
        'settings': {
            'access': {
              'url': AGENCY_URL,
              'auth': {
                  'username': '{{agency_username}}',
                  'password': '{{agency_password}}'
              }
            }
        }
    }
}

with open('image_colorization.red.json', 'w') as f:
    json.dump(red, f, indent=4)