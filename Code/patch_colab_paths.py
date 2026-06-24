import json
import os

colab_notebooks = [
    '01_Colab_DataPrep_and_YOLO_Setup.ipynb',
    '02_Colab_Baseline_YOLOv4.ipynb',
    '03_Colab_Custom_YOLOv4.ipynb',
    '04_Colab_LSTM_FCN_Evaluation.ipynb'
]

def patch_notebook(filename):
    if not os.path.exists(filename):
        return False
        
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                # 1. Fix the absolute path space issue that crashed the local version
                if "os.path.abspath(img_path)" in line:
                    line = line.replace("os.path.abspath(img_path)", "os.path.relpath(img_path, '.')")
                if "os.path.abspath(aug_img_path)" in line:
                    line = line.replace("os.path.abspath(aug_img_path)", "os.path.relpath(aug_img_path, '.')")
                
                # 2. Fix the 23GB Google Drive Pathing issue (use ephemeral disk)
                if "project_path = '/content/drive/MyDrive/cervical_image_grading/Code'" in line:
                    # Replace with ephemeral disk path
                    line = "project_path = '/content/cervical_image_grading/Code'\n"
                
                new_source.append(line)
            
            # Inject copy to drive at the end of training/evaluation notebooks
            if "!python train.py" in "".join(new_source) or "trainer.train()" in "".join(new_source):
                # We append a backup command after training
                if "import shutil" not in "".join(new_source):
                    new_source.append("\n# Backup models to Google Drive after training\n")
                    new_source.append("import shutil\n")
                    new_source.append("import os\n")
                    new_source.append("drive_backup = '/content/drive/MyDrive/Cervical_Models_Backup'\n")
                    new_source.append("os.makedirs(drive_backup, exist_ok=True)\n")
                    new_source.append("if os.path.exists('logs'):\n")
                    new_source.append("    print('Copying model weights to Google Drive...')\n")
                    new_source.append("    !cp -r logs/* \"{drive_backup}/\"\n")
                    new_source.append("    print('Backup complete! Safe to shutdown.')\n")
            
            cell['source'] = new_source
            
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    return True

for nb in colab_notebooks:
    patch_notebook(nb)

print("All Colab notebooks successfully patched for ephemeral pathing and abspath fixes.")
