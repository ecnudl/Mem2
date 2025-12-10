# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Enhanced reward shaping for HotpotQA to address sparse reward problem

def compute_score(solution_str, ground_truth) -> dict:
    """
    Shaped reward function for HotpotQA with intermediate signals.
    Returns dict with multiple reward components.
    """
    from .hotpotqa import last_boxed_only_string, remove_boxed, is_equiv, strip_string

    solution_str_lower = solution_str[-300:].lower()
    ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # Initialize reward components
    rewards = {
        'format_reward': 0.0,      # Has \boxed{}
        'attempt_reward': 0.0,      # Attempts to answer (not "unknown"/"cannot")
        'correct_reward': 0.0,      # Correct answer
        'score': 0.0,               # Final combined score
    }

    # 1. Format reward: +0.2 for using \boxed{} format
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str_lower)
        if string_in_last_boxed is not None:
            rewards['format_reward'] = 0.2
            answer = remove_boxed(string_in_last_boxed)

            # 2. Attempt reward: +0.3 if not giving up
            give_up_phrases = ['no information', 'cannot determine', 'unknown',
                              'not available', 'insufficient', 'unclear']
            if not any(phrase in answer for phrase in give_up_phrases):
                rewards['attempt_reward'] = 0.3

            # 3. Correct reward: +0.5 for correct answer
            for gt in ground_truth_list:
                if is_equiv(answer, gt.lower()):
                    rewards['correct_reward'] = 0.5
                    break
    except Exception as e:
        print(f"Reward shaping error: {e}")

    # Fallback: check for yes/no in last 200 chars
    if rewards['correct_reward'] == 0.0:
        tail = solution_str_lower[-200:]
        yes_idx = tail.rfind("yes")
        no_idx = tail.rfind("no")

        if yes_idx != -1 or no_idx != -1:
            rewards['format_reward'] = max(rewards['format_reward'], 0.1)
            rewards['attempt_reward'] = max(rewards['attempt_reward'], 0.2)

            candidate = "yes" if yes_idx > no_idx else "no"
            for gt in ground_truth_list:
                if is_equiv(candidate, gt.lower()):
                    rewards['correct_reward'] = 0.5
                    break

    # Combine rewards
    rewards['score'] = rewards['format_reward'] + rewards['attempt_reward'] + rewards['correct_reward']

    return rewards
