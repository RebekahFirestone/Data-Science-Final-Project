def run_scripts_in_order(script_list):
    for script in script_list:
        try:
            print(f"Running {script}...")
            with open(script) as file:
                code = file.read()
                exec(code)
            print(f"Finished {script} successfully.\n")
        except Exception as e:
            print(f"Error occurred while running {script}: {e}")
            break

if __name__ == "__main__":
    scripts_to_run = [
        "load_data.py",
        "preprocess.py",
        "build_features.py"
        "train_models.py"
        "evaluate_models.py"
        "visualize.py"
    ]
    run_scripts_in_order(scripts_to_run)
