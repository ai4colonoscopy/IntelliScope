import re
import matplotlib.pyplot as plt
import os

def extract_losses_from_log(log_file_path):
    losses = []

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r"'loss': ([0-9]+\.[0-9]+)", line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)
    
    return losses

def plot_losses(losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

base_path = '/home/projects/u7248002/Project/IntelliScope/cache/checkpoint'
exp_path = 'ColonGPT-phi1.5-siglip-14-7-1-lora-adapter-A2-stg1-2e5-stg2'

log_file_path = os.path.join(base_path, exp_path, f'stdout-{exp_path}.txt')
output_path = os.path.join(base_path, exp_path, 'loss.png')

losses = extract_losses_from_log(log_file_path)
plot_losses(losses, output_path)
