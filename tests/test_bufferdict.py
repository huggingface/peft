import torch

from peft.tuners._buffer_dict import BufferDict


class TestBufferDict:
    def test_init_from_dict_works(self):
        bd = BufferDict(
            {
                "default": torch.randn(10, 2),
            }
        )

    def test_update_from_other_bufferdict(self):
        default_tensor = torch.randn(10, 2)
        non_default_tensor = torch.randn(10, 2)
        bd1 = BufferDict({"default": default_tensor})
        bd2 = BufferDict({"non_default": non_default_tensor})

        bd1.update(bd2)

        assert set(bd1.keys()) == {"default", "non_default"}
        assert torch.allclose(bd1["default"], default_tensor)
        assert torch.allclose(bd1["non_default"], non_default_tensor)

    def test_update_from_dict(self):
        default_tensor = torch.randn(10, 2)
        non_default_tensor = torch.randn(10, 2)
        bd1 = BufferDict({"default": default_tensor})
        d1 = {"non_default": non_default_tensor}

        bd1.update(d1)

        assert set(bd1.keys()) == {"default", "non_default"}
        assert torch.allclose(bd1["default"], default_tensor)
        assert torch.allclose(bd1["non_default"], non_default_tensor)

    def test_update_from_dict_items(self):
        default_tensor = torch.randn(10, 2)
        non_default_tensor = torch.randn(10, 2)
        bd1 = BufferDict({"default": default_tensor})
        d1 = {"non_default": non_default_tensor}

        bd1.update(d1.items())

        assert set(bd1.keys()) == {"default", "non_default"}
        assert torch.allclose(bd1["default"], default_tensor)
        assert torch.allclose(bd1["non_default"], non_default_tensor)
