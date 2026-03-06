#!/usr/bin/env python3
"""Unified doctor tool that runs all checks including continuous training doctor."""
import json
import argparse
import os
import sys
sys.path.append(os.path.dirname(__file__))
import doctor_continuous_training as dct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = {
        'continuous_training': dct.main.__doc__ is not None,
        'checks': None,
    }
    # reuse functions from doctor_continuous_training
    report['checks'] = {
        'repo_list': dct.check_repo_list(),
        'dataset_raw_present': dct.check_dataset_generated(),
        'mlflow_configured': dct.check_mlflow(),
        'gpu_available': dct.check_gpu(),
        'training_loop_pidfile': dct.check_training_loop_alive(),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report['checks'].items():
            print(f"{k}: {v}")


if __name__ == '__main__':
    main()
