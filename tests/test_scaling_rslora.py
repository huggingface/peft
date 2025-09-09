class TestScaling:  
    def test_scaling_simple(self, model):
        n_layers = 5
        rank, lora_alpha = 8, 16
        config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["k_proj"],
        )
        model = get_peft_model(model, config)
        scalings = self.get_scalings(model)
        expected = [lora_alpha / math.sqrt(rank)] * n_layers
        assert scalings == expected
    
        # double
        self.scale_layer(model, 2)
        scalings = self.get_scalings(model)
        expected = [4.0] * n_layers
        assert scalings == expected
    
        # back to original
        self.unscale_layer(model, None)
        scalings = self.get_scalings(model)
        expected = [2.0] * n_layers
        assert scalings == expected
    
        # triple
        self.set_scale(model, "default", 3)
        scalings = self.get_scalings(model)
        expected = [6.0] * n_layers
        assert scalings == expected
    
        # back to original
        self.unscale_layer(model, 3)
        scalings = self.get_scalings(model)
        expected = [2.0] * n_layers
        assert scalings == expected
