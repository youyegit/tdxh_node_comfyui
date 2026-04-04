const { app } = window.comfyAPI.app;

const DYNAMIC_NODE_CONFIG = {
  TdxhKimiDynamicVisionChat: [
    { countWidget: "inputcount", prefix: "image_", type: "IMAGE" },
  ],
  TdxhMultiPlatformDynamicVisionChat: [
    { countWidget: "inputcount", prefix: "image_", type: "IMAGE" },
  ],
  TdxhLocalQwenVLDynamicVisionChat: [
    { countWidget: "inputcount", prefix: "image_", type: "IMAGE" },
  ],
  TdxhLtx23MultimodalDirector: [
    { countWidget: "inputcount", prefix: "image_", type: "IMAGE" },
  ],
  TdxhLtx23AllInOneBridge: [
    { countWidget: "image_inputcount", prefix: "image_", type: "IMAGE" },
    { countWidget: "audio_inputcount", prefix: "audio_", type: "AUDIO" },
    { countWidget: "video_inputcount", prefix: "video_", type: "IMAGE" },
  ],
  TdxhLtx23MultimodalVideoGenerator: [
    { countWidget: "image_inputcount", prefix: "image_", type: "IMAGE" },
    { countWidget: "audio_inputcount", prefix: "audio_", type: "AUDIO" },
    { countWidget: "video_inputcount", prefix: "video_", type: "IMAGE" },
  ],
};

function syncDynamicInputs(node, config) {
  if (!node.inputs) {
    node.inputs = [];
  }

  for (const item of config) {
    const inputCountWidget = node.widgets.find((w) => w.name === item.countWidget);
    const targetNumberOfInputs = inputCountWidget ? Number(inputCountWidget.value) : 1;
    const matchingInputs = node.inputs.filter((input) => input.name && input.name.startsWith(item.prefix));
    const currentCount = matchingInputs.length;

    if (targetNumberOfInputs === currentCount) {
      continue;
    }

    if (targetNumberOfInputs < currentCount) {
      const removableInputs = node.inputs
        .map((input, index) => ({ input, index }))
        .filter(({ input }) => input.name && input.name.startsWith(item.prefix))
        .sort((a, b) => {
          const aNum = parseInt(a.input.name.split("_")[1] || "0", 10);
          const bNum = parseInt(b.input.name.split("_")[1] || "0", 10);
          return bNum - aNum;
        });

      for (let i = 0; i < currentCount - targetNumberOfInputs; i++) {
        const target = removableInputs[i];
        if (target) {
          node.removeInput(target.index);
        }
      }
    } else {
      for (let i = currentCount + 1; i <= targetNumberOfInputs; i += 1) {
        node.addInput(`${item.prefix}${i}`, item.type, { shape: 7 });
      }
    }
  }
}

app.registerExtension({
  name: "tdxh_node_comfyui.dynamic_nodes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const config = DYNAMIC_NODE_CONFIG[nodeData.name];
    if (!config) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated || function () {};
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated.apply(this, arguments);
      this.addWidget("button", "Update inputs", null, () => syncDynamicInputs(this, config));
    };
  },
});
