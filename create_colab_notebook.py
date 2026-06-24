import json
import os

with open('05_Master_Notebook_FIXED.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create the markdown cell
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 12. Export Trained Models & Results (Colab Auto-Download)\n",
        "\n",
        "This cell will automatically zip your trained weights and trigger a download to your local machine. If your Colab session disconnects shortly after this, your results will be safely downloaded."
    ]
}

# Create the code cell
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import shutil\n",
        "try:\n",
        "    from google.colab import files\n",
        "    in_colab = True\n",
        "except:\n",
        "    in_colab = False\n",
        "\n",
        "# Zip the weights directory\n",
        "print('Zipping trained weights...')\n",
        "shutil.make_archive('trained_weights_backup', 'zip', 'Code/trained_weights')\n",
        "print('Zipping complete: trained_weights_backup.zip')\n",
        "\n",
        "if in_colab:\n",
        "    print('Downloading to local machine...')\n",
        "    files.download('trained_weights_backup.zip')\n",
        "else:\n",
        "    print('Not in Colab. File saved locally as trained_weights_backup.zip')\n"
    ]
}

nb['cells'].extend([md_cell, code_cell])

with open('06_Colab_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Created 06_Colab_Training.ipynb successfully.")
