const { app } = window.comfyAPI.app;

app.registerExtension({
  name: "tdxh_node_comfyui.dynamic_nodes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "TdxhKimiDynamicVisionChat" && nodeData.name !== "TdxhMultiPlatformDynamicVisionChat") {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated || function () {};
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated.apply(this, arguments);

      this._type = "IMAGE";
      this.addWidget("button", "Update inputs", null, () => {
        if (!this.inputs) {
          this.inputs = [];
        }

        const inputCountWidget = this.widgets.find((w) => w.name === "inputcount");
        const targetNumberOfInputs = inputCountWidget ? inputCountWidget.value : 1;
        const imageInputs = this.inputs.filter((input) => input.name && input.name.startsWith("image_"));
        const numInputs = imageInputs.length;

        if (targetNumberOfInputs === numInputs) return;

        if (targetNumberOfInputs < numInputs) {
          const inputsToRemove = numInputs - targetNumberOfInputs;
          for (let i = 0; i < inputsToRemove; i++) {
            this.removeInput(this.inputs.length - 1);
          }
        } else {
          for (let i = numInputs + 1; i <= targetNumberOfInputs; ++i) {
            this.addInput(`image_${i}`, this._type, { shape: 7 });
          }
        }
      });
    };
  },
});
