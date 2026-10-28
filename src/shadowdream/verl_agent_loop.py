"""Replay frozen Editor generations through VeRL's V1 agent-loop interface."""
import json

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


class EditorReplayAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, priority=0, **kwargs):
        sessions = json.loads(kwargs["replay_trajectories"])
        session = sessions[int(kwargs["session_id"])]
        return [
            AgentLoopOutput(
                prompt_ids=sample["prompt_ids"],
                response_ids=sample["response_ids"],
                response_mask=[1] * len(sample["response_ids"]),
                response_logprobs=sample["response_logprobs"],
                reward_score=sample["reward"],
                num_turns=1,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "reward_extra_info": {},
                    "min_global_steps": 0,
                    "max_global_steps": 0,
                    "shadowdream_sample_weight": sample["sample_weight"],
                },
            )
            for sample in session
        ]
