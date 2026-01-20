import re

def get_layer_index_from_name(module_name):
    """
    "base_model.model.model.layers.15.self_attn.q_proj" -> 15
    """
    # Pattern looks for 'layers.X.' or 'h.X.' depending on architecture
    match = re.search(r'\.(layers|h|block)\.(\d+)\.', module_name)
    if match:
        return int(match.group(2))
    return None

def get_specific_target_modules(selected_layer_indices, target_suffixes):
    """
    Generates the list of specific module names for Stage 2 Fine-tuning.
    """
    # Format for Llama: "model.layers.{i}.self_attn.{suffix}"
    targets = []
    for i in selected_layer_indices:
        for suffix in target_suffixes:
            targets.append(f"layers\.{i}\..*{suffix}")
    return targets
