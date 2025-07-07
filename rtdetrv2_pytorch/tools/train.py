"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

from mashumaro.mixins.json import DataClassJSONMixin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import neptune
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS


@dataclass
class NeptuneConfiguration(DataClassJSONMixin):
    """
    Specifies Neptune.ai configuration for logging and tracking
    """

    # Neptune project name (ex. "Eyen/car-couplings").
    # Will be set automatically using the model_type if None
    project_name: Optional[str] = None
    # Additional tags describing current run
    tags: List[str] = field(default_factory=lambda: ["RT-DETR2"])

def init_neptune(neptune_configuration: NeptuneConfiguration, model_type: str) -> neptune.Run:
    """Initializes Neptune Run"""
    project_name = neptune_configuration.project_name \
        if neptune_configuration.project_name is not None \
        else f"Eyen/{model_type.replace('_', '-')}"
    os.environ["NEPTUNE_PROJECT"] = project_name
    run = neptune.init_run(project=project_name, tags=neptune_configuration.tags)
    # Export Neptune project name and run ID, so the same run can be accessed from multiple locations (ex. Loggers).
    os.environ["NEPTUNE_CUSTOM_RUN_ID"] = run["sys/id"].fetch()
    return run

def main(args, ) -> None:
    """
    Main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    init_neptune(NeptuneConfiguration(), "car-feature-detector")

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
