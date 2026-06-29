import os
import torch
import torch
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_model_files(path):
    # Create the specific path directory instead of hardcoded ones
    os.makedirs(path, exist_ok=True)
    files = os.listdir(path)
    model_files = []
    for file in files:
        if re.match(r'model_\d+.pth', file):
            model_files.append(file)
    return model_files

def sort_model_files(model_files):
    model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return model_files

def check_model_files(model,x, y, g, stacked_y_g, path, thermal_noise=None, atomic_info=None):
    model_files = get_model_files(path)
    if len(model_files) == 0:
        torch.save(model.state_dict(), path + '/model_' + str(1) + '.pth')
        torch.save(x, path + '/X_' + str(1) + '.pth')
        torch.save(y, path + '/Y_' + str(1) + '.pth')
        torch.save(g, path + '/g_' + str(1) + '.pth')
        torch.save(stacked_y_g, path + '/d_' + str(1) + '.pth')
        if thermal_noise is not None:
            torch.save(torch.tensor(thermal_noise), path + '/thermal_noise_' + str(1) + '.pth')
        if atomic_info is not None:
            torch.save(atomic_info, path + '/atomic_info_' + str(1) + '.pth')
        # Also save in toy model format if this is 2D data (z-component is all zeros)
        if x.shape[-1] == 3 and torch.all(x[:, 2] == 0):
            # This is toy model data - save in 2D format
            positions_2d = x[:, :2]
            forces_2d = g[:, :2] if g.shape[-1] == 3 else g
            torch.save(positions_2d, path + '/positions_' + str(1) + '.pth')
            torch.save(forces_2d, path + '/forces_' + str(1) + '.pth')
            torch.save(y, path + '/energies_' + str(1) + '.pth')
        return False
    else:
        return sort_model_files(model_files)