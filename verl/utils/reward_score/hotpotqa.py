def compute_score(solution_str, ground_truth) -> float:
    # Handle both string and list formats for ground_truth
    if isinstance(ground_truth, str):
        # If ground_truth is a string, split by newlines to handle multi-answer cases
        # For simple cases like "yes", it will be a single-element list
        ground_truth_list = [gt.strip() for gt in ground_truth.split('\n') if gt.strip()]
        if not ground_truth_list:
            # Fallback: use original string if split resulted in empty list
            ground_truth = [ground_truth]
        else:
            ground_truth = ground_truth_list

    def compute_score_single(solution_str, ground_truth) -> float:
        ground_truth = ground_truth.lower()

        retval = 0.
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv(answer, ground_truth):
                    retval = 1.
        except Exception as e:
            print(e)
        # Fallback: if no boxed answer matched, try plain yes/no to avoid 0 reward due to formatting only
        if retval == 0.:
            try:
                # prioritize the last explicit yes/no in the tail
                tail = solution_str[-200:].lower()
                candidate = None
                # simple heuristic: prefer the last occurrence
                yes_idx = tail.rfind("yes")
                no_idx = tail.rfind("no")
                if yes_idx == -1 and no_idx == -1:
                    candidate = None
                elif yes_idx > no_idx:
                    candidate = "yes"
                else:
                    candidate = "no"
                if candidate is not None and is_equiv(
                        candidate,
                        ground_truth,
                ):
                    retval = 1.0
            except Exception:
                pass
        return retval
    solution_str = solution_str[-300:].lower()
    return max(compute_score_single(solution_str, gt) for gt in ground_truth)


# string normalization from
# https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    # remove spaces
    string = string.replace(" ", "")

    return string
