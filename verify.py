import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.dba_tuner_env_environment import DbaTunerEnvironment
from models import DbaTunerAction

def test_reproducibility():
    print("Testing reproducibility...")
    env1 = DbaTunerEnvironment()
    obs1_1 = env1.reset(level=1, seed=42)
    plan1_1 = env1._baseline_plan_cost
    
    env2 = DbaTunerEnvironment()
    obs2_1 = env2.reset(level=1, seed=42)
    plan2_1 = env2._baseline_plan_cost
    
    assert plan1_1 == plan2_1, f"Reproducibility failed: {plan1_1} != {plan2_1}"
    print(f"Reproducibility passed! Baseline cost: {plan1_1}")
    env1.close()
    env2.close()

def test_off_by_one_and_done():
    print("Testing off-by-one and done...")
    env = DbaTunerEnvironment()
    obs = env.reset(level=1, seed=42)
    assert not obs.done
    
    tables = ["users", "orders"]
    for i in range(env._max_steps):
        a = DbaTunerAction(action_type="get_stats", table=tables[i % 2])
        obs = env.step(a)
    
    assert obs.done, "Episode should be done after max steps"
    
    # Take one more step to see the specific max steps error
    obs_over = env.step(DbaTunerAction(action_type="explain"))
    assert obs_over.done, "Should remain done"
    assert "Episode is already done" in obs_over.error_message or "Max steps" in obs_over.error_message, f"Got error: {obs_over.error_message}"
    print("Off-by-one passed!")
    env.close()

def test_level_4_rejection():
    print("Testing Level 4 rewrite rejection...")
    env = DbaTunerEnvironment()
    env.reset(level=4, seed=42)
    a = DbaTunerAction(action_type="rewrite", new_sql="SELECT * FROM orders")
    obs = env.step(a)
    assert "Level 4 is an index-budget-only task" in obs.error_message, f"Got: {obs.error_message}"
    print("Level 4 rewrite rejection passed!")
    env.close()

def test_task_solved_gating():
    print("Testing task_solved gating...")
    env = DbaTunerEnvironment()
    env.reset(level=1, seed=42)
    
    # 1. Earn explain bonus (reward +0.1)
    obs1 = env.step(DbaTunerAction(action_type="explain"))
    assert obs1.metadata["reasoning_bonus_paid"]
    
    # 2. Call done
    obs2 = env.step(DbaTunerAction(action_type="done"))
    assert obs2.done
    # Terminal reward must be 0 because task_solved is False!
    assert obs2.reward == 0.0, f"Reward should be 0.0, got {obs2.reward}"
    print("Task solved gating passed!")
    env.close()

if __name__ == "__main__":
    test_reproducibility()
    test_off_by_one_and_done()
    test_level_4_rejection()
    test_task_solved_gating()
    print("All tests passed!")
