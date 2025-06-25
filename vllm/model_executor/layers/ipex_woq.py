try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)

    # @CustomOp.register("ipex_woq_linear")
    # class IPEXWoqLinear(CustomOp):
    def __init__(
        self,
        layer,
        qconfig,
        bias,
        group_size,
        quant_method: int,
    ):
        super().__init__()
        linear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.qweight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            bias=bias,
            group_size=group_size,
            quant_method=quant_method  # type: ignore
        )
        self.linear = linear

    def forward_native(self, x):
        return torch.empty((x.size(0), self.linear.out_features),
                           dtype=x.dtype)

    def forward_xpu(self, x):
        return self.linear(x)
