import json

with open('06_Colab_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 0. Colab Setup & Dataset Download\n",
        "Run this cell first! Make sure you drag and drop your `kaggle.json` file directly into the Colab file explorer on the left before running this."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "\n",
        "# 1. Setup Kaggle credentials\n",
        "!mkdir -p ~/.kaggle\n",
        "!cp kaggle.json ~/.kaggle/\n",
        "!chmod 600 ~/.kaggle/kaggle.json\n",
        "\n",
        "# 2. Install Kaggle CLI\n",
        "!pip install -q kaggle\n",
        "\n",
        "# 3. Download the Dataset\n",
        "# REPLACE 'your-dataset-name' below with the actual name of your Kaggle dataset!\n",
        "dataset_identifier = 'ahmadfauziramadhan/your-dataset-name'\n",
        "zip_name = dataset_identifier.split('/')[1] + '.zip'\n",
        "\n",
        "!kaggle datasets download -d $dataset_identifier\n",
        "\n",
        "# 4. Unzip directly into the Code/sipakmed_data folder\n",
        "!unzip -q $zip_name -d Code/sipakmed_data\n",
        "\n",
        "print('Dataset successfully loaded and ready for training!')\n"
    ]
}

nb['cells'] = [md_cell, code_cell] + nb['cells']

with open('06_Colab_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Injected Kaggle setup into 06_Colab_Training.ipynb")
