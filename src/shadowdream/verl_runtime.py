"""Small VeRL V1 runner for grouped Editor replay training."""
from pathlib import Path

import hydra
import ray
import torch
import transfer_queue as tq
import verl
from tensordict import TensorDict
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync
from verl.utils.config import validate_config
from verl.workers.utils.padding import response_to_nested


class ShadowDreamTrainer(PPOTrainerSync):
    def _compute_advantage(self, batch, metrics):
        batch = super()._compute_advantage(batch, metrics)
        if not self.config.distillation.enabled:
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["advantages", "response_mask", "extra_fields"],
            )
            trajectory_ids = {key.rsplit("_", 1)[0] for key in batch.keys}
            scales = torch.tensor(
                [
                    item["shadowdream_sample_weight"] * len(batch.keys) / len(trajectory_ids)
                    for item in data["extra_fields"]
                ],
                dtype=torch.float32,
            )
            advantages = data.to_padded_tensor()["advantages"]
            weighted = advantages * scales.to(advantages.device).unsqueeze(-1)
            tq.kv_batch_put(
                keys=batch.keys,
                partition_id=batch.partition_id,
                fields=TensorDict(
                    {"advantages": response_to_nested(weighted, data["response_mask"])},
                    batch_size=len(batch),
                ),
            )
            metrics["shadowdream/trajectory_count"] = len(trajectory_ids)
            metrics["shadowdream/model_call_count"] = len(batch)
        return batch


@ray.remote
class ShadowDreamTaskRunner:
    def run(self, config):
        config.transfer_queue.enable = True
        tq.init(config.transfer_queue)
        try:
            trainer = ShadowDreamTrainer(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
        finally:
            tq.close()


@hydra.main(
    config_path=str(Path(verl.__file__).parent / "trainer" / "config"),
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    validate_config(config=config, use_reference_policy=False, use_critic=False)
    run_ppo(config, task_runner_class=ShadowDreamTaskRunner)


if __name__ == "__main__":
    main()
