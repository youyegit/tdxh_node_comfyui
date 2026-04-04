const { app } = window.comfyAPI.app;

app.registerExtension({
  name: "tdxh_node_comfyui.dynamic_nodes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (
      nodeData.name !== "TdxhKimiDynamicVisionChat" &&
      nodeData.name !== "TdxhMultiPlatformDynamicVisionChat" &&
      nodeData.name !== "TdxhLocalQwenVLDynamicVisionChat"
    ) {
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
            const removableImageInputs = this.inputs
              .map((input, index) => ({ input, index }))
              .filter(({ input }) => input.name && input.name.startsWith("image_"))
              .sort((a, b) => {
                const aNum = parseInt(a.input.name.split("_")[1] || "0", 10);
                const bNum = parseInt(b.input.name.split("_")[1] || "0", 10);
                return bNum - aNum;
              });
            const target = removableImageInputs[i];
            if (target) {
              this.removeInput(target.index);
            }
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
