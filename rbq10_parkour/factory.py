"""Copy the manufacturer's ONNX Gemm/ELU weights without changing the policy."""
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
import torch
from torch import nn
from .contract import ASSETS


def load_factory_policy(device="cpu", verify=False):
    graph = onnx.load(str(ASSETS / "official/policy.onnx"))
    onnx.checker.check_model(graph)
    if [node.op_type for node in graph.graph.node] != ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
        raise ValueError("Manufacturer policy graph differs from the verified Gemm/ELU model")
    layers = []
    sizes = (45, 512, 256, 128, 12)
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < 3:
            layers.append(nn.ELU())
    model = nn.Sequential(*layers).eval()
    arrays = {p.name.removeprefix("actor."): numpy_helper.to_array(p).copy() for p in graph.graph.initializer}
    model.load_state_dict({key: torch.from_numpy(arrays[key]) for key in model.state_dict()}, strict=True)
    maximum_error = None
    if verify:
        reference = ReferenceEvaluator(graph)
        rng = np.random.default_rng(2026)
        maximum_error = 0.0
        with torch.inference_mode():
            for _ in range(32):
                obs = rng.normal(size=(1, 45)).astype(np.float32)
                target = reference.run(None, {graph.graph.input[0].name: obs})[0]
                actual = model(torch.from_numpy(obs)).numpy()
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - target))))
        if maximum_error >= 1e-4:
            raise ValueError(f"Manufacturer ONNX conversion numerical mismatch: {maximum_error}")
    return model.to(device).requires_grad_(False), maximum_error
