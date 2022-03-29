import sys

import wandb

import workflow


def run_agent_by_sweep_id(sweep_id):
    wandb.login()
    wandb.agent(sweep_id=sweep_id,
                function=workflow.train,
                )


if __name__ == '__main__':
    sweep_id = sys.argv[1]
    run_agent_by_sweep_id(sweep_id)
