"""Print the rope parameters vLLM builds for a model with a given --hf-overrides JSON, without loading weights.

Run it with the image's python, inside a serving container or a throwaway one:
    python3 ropecheck.py '{"rope_scaling":{...}}' [/app/models/Mistral-Large-Instruct-2411]
Check that "base" is the model's rope_theta (1e6 for Mistral Large), not 10000.
"""
import json, sys
from vllm.engine.arg_utils import EngineArgs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

ov = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
model = sys.argv[2] if len(sys.argv) > 2 else "/app/models/Mistral-Large-Instruct-2411"
ea = EngineArgs(model=model, hf_overrides=ov, max_model_len=131072, tokenizer_mode="mistral")
hc = ea.create_model_config().hf_text_config
print("rope_parameters:", getattr(hc, "rope_parameters", None))
with set_current_vllm_config(VllmConfig()):
    r = get_rope(hc.head_dim, max_position=1048576 if ov else 131072, is_neox_style=True,
                 rope_parameters=hc.rope_parameters)
print(type(r).__name__, {k: getattr(r, k, None) for k in ("base", "scaling_factor", "mscale", "max_position_embeddings")})
