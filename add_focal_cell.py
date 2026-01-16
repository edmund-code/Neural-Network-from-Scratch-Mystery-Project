import json

notebook_path = "MLP_classifiers.ipynb"

# New cell content for Focal Loss experiment
new_cell_source = [
    "# -----------------------------\n",
    "# 8) Run Focal Loss Comparison\n",
    "# -----------------------------\n",
    "print(\"\\n=== Model 6: Focal Loss (gamma=2.0) ===\")\n",
    "\n",
    "# Note: ensuring we use the same dataset object as previous runs\n",
    "# If train_loader is not defined, ensure the dataset is loaded.\n",
    "\n",
    "y_true_focal, y_pred_focal = cross_validate_mlp_standardscaler(\n",
    "    dataset=train_loader.dataset,\n",
    "    n_splits=5,\n",
    "    batch_size=128,\n",
    "    random_state=0,\n",
    "    epochs=40,\n",
    "    hidden=(256, 128),     # match Model 5\n",
    "    dropout=0.40,          # match Model 5\n",
    "    weight_decay=5e-4,     # match Model 5\n",
    "    patience=5,            # match Model 5\n",
    "    min_delta=1e-4,\n",
    "    loss_fn_name=\"focal\",\n",
    "    focal_gamma=2.0,\n",
    ")"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Model 6: Focal Loss (gamma=2.0)\n",
        "\n",
        "Comparing Focal Loss against the Cross-Entropy baseline (Model 5) using the same regularization and early stopping settings."
    ]
}

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    
    # Try to find the position after "Model 5" output
    insert_idx = -1
    for i, cell in enumerate(cells):
        source = "".join(cell.get('source', []))
        if "Model 5: light regularization" in source:
            # We found the Model 5 markdown/code.
            # Usually the next cell is the code, then maybe output.
            # We want to insert AFTER the code cell that runs Model 5.
            # Let's look for the code cell that invokes cross_validate_mlp_standardscaler
            # subsequent to this markdown.
            for j in range(i+1, len(cells)):
                c_source = "".join(cells[j].get('source', []))
                if "cross_validate_mlp_standardscaler" in c_source and "loss_fn_name" not in c_source: 
                    # Assuming Model 5 didn't explicitly set loss_fn_name="ce" (it's default)
                    # or it set it to "ce"
                    insert_idx = j + 1
                    break
            break
            
    if insert_idx == -1:
        # Fallback: Append to the end
        print("Model 5 location not found, appending to end.")
        cells.append(markdown_cell)
        cells.append(new_cell)
    else:
        print(f"Inserting after cell {insert_idx}")
        # Insert markdown then code
        cells.insert(insert_idx, new_cell)
        cells.insert(insert_idx, markdown_cell)

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook updated successfully.")

except Exception as e:
    print(f"Error: {e}")
