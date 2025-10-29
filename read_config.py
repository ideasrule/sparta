
import shlex



# =============================================================================
# 1) CONFIG‐FILE READER
# =============================================================================

def read_config(path):
    """
    Read a parameter‐configuration file of the format you showed.  

    Each non‐comment/non‐blank line must have either:
      (a) 3 tokens:  Name, Value, Free?                 (for fixed/independent),
      (b) 6 tokens:  Name, Value, Free?, PriorPar1, PriorPar2, PriorType  (for free).

    Free? ∈ {'free','fixed','independent'}.
    PriorType ∈ {'U','LU','N'}.

    Returns
    -------
    param_info : dict
      { param_name: {
            'value': float or str,
            'free' : one of above six strings,
            'prior': None  (if fixed/independent),
                     OR { 'type': 'U'|'LU'|'N', 'p1': float, 'p2': float }
        }
      }
    """
    param_info = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            tokens = shlex.split(line)
            if len(tokens) < 3:
                raise ValueError(f"Each non‐comment line must have ≥3 tokens. Got:\n  {line}")

            name  = tokens[0]
            value = tokens[1]
            free  = tokens[2].strip("'\"")

            if free not in ("free", "fixed", "independent"):
                raise ValueError(f"Invalid Free? flag '{free}' on line:\n  {line}")

            entry = {"value": None, "free": free, "prior": None}

            # Try to parse the “Value” as float, otherwise keep as string:
            try:
                val = float(value)
            except ValueError:
                entry["value"] = value.strip("'\"")
            else:
                entry["value"] = val

            # If it’s “free” or “white_free”, we expect exactly 6 tokens total:
            if free in ("free", "white_free"):
                if len(tokens) != 6:
                    raise ValueError(
                        f"Parameter '{name}' is marked '{free}', so we need 6 columns "
                        f"(Name,Value,Free?,PriorPar1,PriorPar2,PriorType), but got:\n  {line}"
                    )
                try:
                    p1 = float(tokens[3])
                    p2 = float(tokens[4])
                except ValueError:
                    raise ValueError(f"Could not parse PriorPar1/PriorPar2 as floats for '{name}':\n  {line}")

                prior_type = tokens[5]
                if prior_type not in ("U", "LU", "N"):
                    raise ValueError(f"Unknown PriorType '{prior_type}' for '{name}'. Must be U, LU, or N.")

                entry["prior"] = {"type": prior_type, "p1": p1, "p2": p2}

            else:
                # fixed / white_fixed / shared / independent ⇒ no prior columns
                if len(tokens) != 3:
                    raise ValueError(
                        f"Parameter '{name}' is marked '{free}', so it must have exactly 3 tokens. "
                        f"Line has {len(tokens)} tokens:\n  {line}"
                    )
                entry["prior"] = None

            param_info[name] = entry

    return param_info


def get_free_params(param_info):
    """
    Return a dict { param_name: initial_value } for all parameters
    whose 'free' flag is 'free' or 'white_free'.
    """
    free_dict = {}
    for name, info in param_info.items():
        if info["free"] in ("free", "white_free"):
            free_dict[name] = info["value"]
    return free_dict


def get_fixed_params(param_info):
    """
    Return a dict { param_name: value } for all parameters whose 'free' flag
    is in {'fixed','independent'}.
    """
    fixed_dict = {}
    for name, info in param_info.items():
        if info["free"] in ("fixed","independent"):
            fixed_dict[name] = info["value"]
    return fixed_dict


def get_priors(param_info):
    """
    Return a dict { param_name: prior_spec }, where prior_spec is either None
    (if no prior) or { 'type': 'U'|'LU'|'N', 'p1': float, 'p2': float }.
    """
    priors = {}
    for name, info in param_info.items():
        priors[name] = info["prior"]
    return priors
