import torch
from hrm_model import HRMChess

# Betöltjük a modellt
model = HRMChess(hidden_dim=256, N=8, T=8)
checkpoint = torch.load('best_hrm_chess_model_1411.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy bemenet: [batch, 30, 8, 8]
dummy_input = torch.randn(1, 30, 8, 8)

# Exportálás ONNX formátumba
onnx_path = 'hrm_model.onnx'
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f'✅ ONNX exportálva: {onnx_path}')
