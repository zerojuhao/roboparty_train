"""CPU-only IPC fixture for the persistent evaluator tests."""

import json
import os
import sys

import torch

request = torch.load(sys.argv[sys.argv.index('--tracking_eval_request') + 1], weights_only=False)
rounds = 0
for line in sys.stdin:
    message = json.loads(line)
    data = torch.load(message['request_path'], weights_only=False)
    rounds += 1
    result = {'global_step': data['global_step'], 'worker_pid': os.getpid(),
              'rounds': rounds, 'value': data['model_state']['value'].item()}
    path = data['result_path']
    torch.save(result, path + '.tmp')
    os.replace(path + '.tmp', path)
