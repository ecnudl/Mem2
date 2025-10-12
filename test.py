import pkgutil
print("ray installed:", pkgutil.find_loader("ray") is not None)
import pkgutil
print("vllm installed:", pkgutil.find_loader("vllm") is not None)
import pkgutil
print("flash_attn installed:", pkgutil.find_loader("flash_attn") is not None)