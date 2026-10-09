import csv
import os
baseline_mmau = {"Qwen2-Audio-7B-Instruct": 0.42}

with open("evaluation/aggregated_results.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        Gradient_reversal_alpha,Learning_rate,LLM_loss_weight,Model,Checkpoint_epoch,Checkpoint_step,accuracy_gender_detection,_,_,accuracy_mmau = row
        accuracy_mmau = float(accuracy_mmau)
        if Model in baseline_mmau and accuracy_mmau < baseline_mmau[Model] or float(accuracy_gender_detection) > 0.6:
            checkpoint = f"adversarial/logs/bs=8,cs=125,ds=180,e=1,ebs=24,gra={Gradient_reversal_alpha},kf=False,ls=0.05,lr={Learning_rate},llw={LLM_loss_weight},mn={Model},ns=4000,wd=0.005/checkpoint/checkpoint_{Checkpoint_epoch}_{Checkpoint_step}.pt"
            if os.path.exists(checkpoint):
                print(f"Removing {checkpoint} with gender detection accuracy {accuracy_gender_detection} and MMaU {accuracy_mmau}")
                os.remove(checkpoint)
