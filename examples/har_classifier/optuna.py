from __future__ import annotations

import optuna
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
OPTUNA_LOGS = LOGS/ "optuna_logs"

def objective(trial) -> int| float:
    trial_number = trial.number
    delta_reduction = trial.suggest_int("delta_reduction", 1, 4, step=1)
    learning_rate = trial.suggest_int("learning_rate", 1, 4, step=1)
    momentum = trial.suggest_int("momentum", 1, 4, step=1)
    rounding_mode = 0 # HALF_AWAY
    epochs = 20
    batch = 64



    result = subprocess.run(
        [
            './build/examples/examples/har_classifier/train_c_har_classifier',
            str(trial_number),
            str(delta_reduction),
            str(learning_rate),
            str(momentum),
            str(rounding_mode),
            str(epochs),
            str(batch)
        ],
        capture_output=True,
        text=True
    )
    prefix = 'examples/har_classifier/logs/trial'
    with open(prefix + trial_number + '.json', 'r') as f:
        data = json.load(f)
        yield data

    test_loss = data.get("final", {}).get("test_loss")
    test_acc = data.get("final", {}).get("test_acc")

    #test_loss = data["final"]["test_loss"]
    #test_acc = data["final"]["test_acc"]
    #print('FINAL test_loss=' + test_loss + '  test_acc=' + test_acc )
    #optuna_element{
    #    "optuna": [
    #        {"name": "Max"},
    #        {"name": "Lisa"}
    #    ]
    #}
    #data["items"].insert(3, neues_element)



    trial.set_user_attr("delta_reduction", delta_reduction)
    trial.set_user_attr("lr", learning_rate)
    trial.set_user_attr("momentum", momentum)
    trial.set_user_attr("rounding_mode", rounding_mode)

    trial.set_user_attr("test_loss", test_loss)
    trial.set_user_attr("test_acc", test_acc)

    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("batch", batch)

    return test_acc


def main():
    optuna_results_dir = OPTUNA_LOGS
    optuna_results_dir.mkdir(parents=True, exist_ok=True)

    study_name = "binary_classifier_train_c"
    study_db_path: Path = optuna_results_dir / f"{study_name}.db"

    study = optuna.create_study(
        study_name = study_name,
        direction="maximize",
        storage = f"sqlite:///{study_db_path.resolve()}",
        load_if_exists=True)

    study.optimize(objective, n_trials=30)
    space = intersection_search_space(study.get_trials())
    print(f"Intersection Search Space Of Trials : {space}")

    print(f"Best Trial Number : {study.best_trial.number}")
    print(f"Best Trial Value : {study.best_trial.value}")
    print(f"Best Trial Params : {study.best_trial.params}")
    print(f"Best Value : {study.best_value}")
    print(f"Best Params : {study.best_params}")

if __name__ == "__main__":
    main()
