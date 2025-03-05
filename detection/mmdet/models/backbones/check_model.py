from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from classification.models import *
model = multiscalelgtformer_tiny()
model.eval()
flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
print(flop_count_table(flops))
