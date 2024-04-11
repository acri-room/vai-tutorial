# https://github.com/Xilinx/Vitis-AI/tree/v3.5/examples/vai_runtime/resnet50_mt_py

"""
Copyright 2022-2023 Advanced Micro Devices Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
from omegaconf import DictConfig

import torch
import torchvision
import numpy as np
import timg

import xir
import vart

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(cfg: DictConfig):
    # Load xmodel
    graph = xir.Graph.deserialize(cfg.xmodel)
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1 # only one DPU kernel
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    # Get input/output tensor information
    input_tensors = runner.get_input_tensors()
    #print(input_tensors[0].dims) # 8, 32, 32, 3
    output_tensors = runner.get_output_tensors()
    #print(output_tensors[0].dims) # 8, 10

    # Check batch size
    batch_size = cfg.data.batch_size
    assert batch_size <= input_tensors[0].dims[0]

    # Calculate input scale for preprocessing
    input_fixpos = runner.get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    # Calculate output scale for postprocessing
    output_fixpos = runner.get_output_tensors()[0].get_attr("fix_point")
    output_scale = 2**output_fixpos

    # Allocate input/output buffer
    input_data = [np.empty(input_tensors[0].dims, dtype=np.int8, order="C")]
    output_data = [np.empty(output_tensors[0].dims, dtype=np.int8, order="C")]

    # Instanciate data module
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    for i, (x, y) in enumerate(datamodule.test_dataloader()):

        if i == cfg.num_batches:
            break

        # Preprocess input data
        input_data[0][0:batch_size, :] = torch.permute(x, (0, 2, 3, 1)) * input_scale

        # Run inference on the device
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)

        # Postprocess output data
        output = torch.from_numpy(output_data[0].astype(np.float32)).clone()
        output = torch.softmax(output / output_scale, dim=1)

        # Inverse of normalization
        torchvision.transforms.v2.functional.normalize(x, (0.0, 0.0, 0.0), (1/0.2023, 1/0.1994, 1/0.2010), inplace=True)
        torchvision.transforms.v2.functional.normalize(x, (-0.4914, -0.4822, -0.4465), (1.0,1.0,1.0), inplace=True)

        # Print result
        topk = torch.topk(output, 3, dim=1)
        for batch in range(batch_size):
            # Print image
            renderer = timg.Renderer()
            renderer.load_image(torchvision.transforms.functional.to_pil_image(x[batch]))
            renderer.render(timg.METHODS['a24h']['class'])

            # Print Top-K
            for j in range(topk.values[batch].shape[0]):
                print("%d: %5.1f%% %s %s" % (j,
                    topk.values[batch][j] * 100,
                    CIFAR10_LABELS[topk.indices[batch][j]],
                    "*** correct label ***" if topk.indices[batch][j] == y[batch] else ""))

            print()

if __name__ == "__main__":
    main()
