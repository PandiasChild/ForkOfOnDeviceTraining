from __future__ import annotations

import logging
import optuna
import subprocess
import string
from pathlib import Path
import json
import requests
import time
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parents[1]
HAR_CLASSIFIER_LOGS = PROJECT_ROOT/ "examples" / "har_classifier" / "logs"
OPTUNA_LOGS = HAR_CLASSIFIER_LOGS / "optuna_logs"
DELTA_REDUCTION = int(sys.argv[3])
ID = str(sys.argv[2])
SEED = int(sys.argv[1])
STUDY_NAME = "study_" + str(ID) + "reduce" + str(DELTA_REDUCTION)

def send_notification(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        requests.post(url, data=data)
    except:
        None
def _objective_impl(trial):
    trial_number = trial.number
    #delta_reduction = trial.suggest_int("delta_reduction", 1, 4, step=1)
    learning_rate = trial.suggest_float("learning_rate", 0.000001, 0.0001, step=0.000001) #0.001 = 1e-3 & 1e-5 = 0.00001
    momentum = trial.suggest_float("momentum", 0.7, 0.95, step=0.000001) #0.9
    # rounding_mode = 0 # HALF_AWAY
    epochs = 50
    batch = 64 # möchte ich klein haben, weil für embedded device

    trial.set_user_attr("delta_reduction", DELTA_REDUCTION)
    trial.set_user_attr("lr", learning_rate)
    trial.set_user_attr("momentum", momentum)
    #trial.set_user_attr("rounding_mode", rounding_mode)

    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("batch", batch)

    test_loss_delta = 1
    test_acc_delta = 0
    #start = time.time()
    try:
        result = subprocess.run(
            [
                './build/examples_memprofile/examples/har_classifier/train_c_har_classifier_delta',
                str(trial_number),
                str(DELTA_REDUCTION),
                str(learning_rate),
                str(momentum),
                str(epochs),
                str(batch),
                STUDY_NAME,
                SEED
                #str(rounding_mode)
            ],
            check = True,
            cwd=PROJECT_ROOT,
            #capture_output=True,
            text=True
        )
        #print("C fertig nach", time.time() - start, "Sekunden")
        #print("Returncode:", result.returncode)
        #print("Output:", result.stdout)
        test_duration_delta = 0
        prefix = HAR_CLASSIFIER_LOGS / "with_deltas" / f"delta_reduction_{DELTA_REDUCTION}"
        json_path = Path(str(prefix) + "trial_" + str(trial_number) + ".json")
        print(f"[Trial {trial_number}] erwarteter json_path: {json_path}")
        print(f"[Trial {trial_number}] existiert diese Datei? {json_path.exists()}")
        print(f"[Trial {trial_number}] Verzeichnisinhalt: {list(json_path.parent.iterdir()) if json_path.parent.exists() else 'Verzeichnis existiert nicht'}")

        with open(json_path, 'r') as f:
            data = json.load(f)

            for epochs in data["epochs"]:
                test_duration_delta += epochs.get("wall_s")

            test_loss_delta = data.get("final", {}).get("test_loss")
            test_acc_delta = data.get("final", {}).get("test_acc")

            trial.set_user_attr("test_loss_delta", test_loss_delta)
            trial.set_user_attr("test_acc_delta", test_acc_delta)
            trial.set_user_attr("test_duration_delta", test_duration_delta)

    except subprocess.CalledProcessError as e:
        if(e.returncode == 5):
            trial.set_user_attr("error", "matmulSymInt32TensorsWithBias")
        if(e.returncode == 6):
            trial.set_user_attr("error", "rescaleIntoAccumulatorScale")
        if(e.returncode == 2):
            trial.set_user_attr("error", "GATES FAILED")
        prefix = HAR_CLASSIFIER_LOGS / "with_deltas" / f"delta_reduction_{DELTA_REDUCTION}"
        json_path = Path(str(prefix) + "trial_" + str(trial_number) + ".json")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                initial_val_loss = None
                initial_val_acc = None
                if isinstance(data.get("epochs"), list) and data["epochs"]:
                    initial_val_loss = data["epochs"][0].get("initial_val_loss")
                    initial_val_acc = data["epochs"][0].get("initial_val_acc")

                trial.set_user_attr("initial_val_loss", initial_val_loss)
                trial.set_user_attr("initial_val_acc", initial_val_acc)
        except (FileNotFoundError, json.JSONDecodeError):
            trial.set_user_attr("json_error", True)
            return test_acc_delta, test_loss_delta
        return test_acc_delta, test_loss_delta

    except FileNotFoundError as e:
        try:
            with open('telegram_bot.json', 'r') as f:
                telegram_bot = json.load(f)

                bot_token = telegram_bot.get("BOT_TOKEN", {})
                chat_id = telegram_bot.get("CHAT_ID", {})
                message = f"Training DELTA fehlgeschlagen:\ntrial_number {trial_number}\nPython oder das Skript wurde nicht gefunden: {e}\n"
                send_notification(bot_token, chat_id, message)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        trial.set_user_attr("error", "json_error")
        return test_acc_delta, test_loss_delta
    #---------------------------------------------------------------------------------------------------------------
    try:
        result = subprocess.run(
            [
                './build/examples_memprofile/examples/har_classifier/train_c_har_classifier_sym',
                str(trial_number),
                str(DELTA_REDUCTION),
                str(learning_rate),
                str(momentum),
                str(epochs),
                str(batch),
                #str(rounding_mode)
            ],
            check=True,
            cwd=PROJECT_ROOT,
            #capture_output=True,
            text=True
        )
        test_duration_sym = 0
        prefix = HAR_CLASSIFIER_LOGS / "without_deltas" / f"delta_reduction_{DELTA_REDUCTION}"
        json_path = Path(str(prefix) + "trial_" + str(trial_number) + ".json")
        with open(json_path, 'r') as f:
            data_sym = json.load(f)

            for epochs in data_sym["epochs"]:
                test_duration_sym += epochs.get("wall_s")

            test_loss_sym = data_sym.get("final", {}).get("test_loss")
            test_acc_sym = data_sym.get("final", {}).get("test_acc")

            trial.set_user_attr("test_loss_sym", test_loss_sym)
            trial.set_user_attr("test_acc_sym", test_acc_sym)
            trial.set_user_attr("test_duration_sym", test_duration_sym)

    except subprocess.CalledProcessError as e:
        if(e.returncode == 5):
            trial.set_user_attr("error", "matmulSymInt32TensorsWithBias")
        if(e.returncode == 6):
            trial.set_user_attr("error", "rescaleIntoAccumulatorScale")
        if(e.returncode == 2):
            trial.set_user_attr("error", "GATES FAILED")
        try:
            prefix = HAR_CLASSIFIER_LOGS / "without_deltas" / f"delta_reduction_{DELTA_REDUCTION}"
            json_path = Path(str(prefix) + "trial_" + str(trial_number) + ".json")

            with open(json_path, 'r') as f:
                data = json.load(f)

                initial_val_loss = None
                initial_val_acc = None
                if isinstance(data.get("epochs"), list) and data["epochs"]:
                    initial_val_loss = data["epochs"][0].get("initial_val_loss")
                    initial_val_acc = data["epochs"][0].get("initial_val_acc")

                trial.set_user_attr("initial_val_loss", initial_val_loss)
                trial.set_user_attr("initial_val_acc", initial_val_acc)
        except (FileNotFoundError, json.JSONDecodeError):
            trial.set_user_attr("error", "json_error")
            return test_acc_delta, test_loss_delta
        return test_acc_delta, test_loss_delta

    except FileNotFoundError as e:
        try:
            with open('telegram_bot.json', 'r') as f:
                telegram_bot = json.load(f)

                bot_token = telegram_bot.get("BOT_TOKEN", {})
                chat_id = telegram_bot.get("CHAT_ID", {})
                message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nPython oder das Skript wurde nicht gefunden: {e}\n"
                send_notification(bot_token, chat_id, message)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        trial.set_user_attr("error", "json_error")
        return test_acc_delta, test_loss_delta

    trial.set_user_attr("error", "NO ERROR")
    return test_acc_delta, test_loss_delta
''' 
DELTA
    except OSError as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nBetriebssystemfehler: {e}\n"
            send_notification(bot_token, chat_id, message)

    

    except PermissionError:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training DELTA fehlgeschlagen:\ntrial_number {trial_number}\nKeine Berechtigung zum Ausführen: {e}\n"
            send_notification(bot_token, chat_id, message)

    except TimeoutError:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training DELTA fehlgeschlagen:\ntrial_number {trial_number}\nZeitüberschreitung: {e}\n"
            send_notification(bot_token, chat_id, message)

    except Exception as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training DELTA fehlgeschlagen:\ntrial_number {trial_number}\nexception: {e}\n"
            send_notification(bot_token, chat_id, message)
------------------------------------------------------------------------------------------
SYM
    except OSError as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nBetriebssystemfehler: {e}\n"
            send_notification(bot_token, chat_id, message)
    except PermissionError:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nKeine Berechtigung zum Ausführen: {e}\n"
            send_notification(bot_token, chat_id, message)

    except TimeoutError:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nZeitüberschreitung: {e}\n"
            send_notification(bot_token, chat_id, message)

    except Exception as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nexception: {e}\n"
            send_notification(bot_token, chat_id, message)
            '''

def objective(trial):
    try:
        return _objective_impl(trial)
    except Exception:
        import traceback
        crash_log_path = OPTUNA_LOGS / f"{STUDY_NAME}_objective_crashes.log"
        try:
            with open(crash_log_path, "a") as f:
                f.write(f"--- Trial {trial.number} ---\n")
                f.write(traceback.format_exc())
                f.write("\n")
        except Exception as e:
            print(f"[Trial {trial.number}] Konnte Crash nicht loggen: {e}")
            print(traceback.format_exc())
        return 0, 1
def main():
    optuna_results_dir = OPTUNA_LOGS
    optuna_results_dir.mkdir(parents=True, exist_ok=True)

    # Create a file handler
    file_handler = logging.FileHandler(str(optuna_results_dir) + "/optuna.log")
    file_handler.setLevel(logging.INFO)

    # Add it to Optuna's logger
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.addHandler(file_handler)

    # Optional: keep console quiet
    #optuna.logging.disable_default_handler()


    study_db_path: Path = optuna_results_dir / f"{STUDY_NAME}.db"

    study = optuna.create_study(
        study_name = STUDY_NAME,
        directions=["maximize", "minimize"],
        storage = f"sqlite:///{study_db_path.resolve()}",
        load_if_exists=True)

    study.optimize(objective, n_trials=200, n_jobs = 1, catch=(Exception,))
    #space = intersection_search_space(study.get_trials())

    #fig = plot_optimization_history(study)
    #fig.write_html("optimization_history.html")
    #fig.write_image("optimization_history.png")

    # Optimierungsverlauf
    #plot_optimization_history(study).show()

    # Wichtigkeit der Hyperparameter
    #plot_param_importances(study).show()

    # Parallel Coordinates
    #plot_parallel_coordinate(study).show()

    # Slice Plot
    #plot_slice(study).show()

    # Konturdiagramm
    #plot_contour(study).show()

    # Empirical Distribution Function
    #plot_edf(study).show()

    #print(f"Intersection Search Space Of Trials : {space}")

    #print(f"Best Trial Number : {study.best_trial.number}")
    #print(f"Best Trial Value : {study.best_trial.value}")
    #print(f"Best Trial Params : {study.best_trial.params}")
    #print(f"Best Value : {study.best_value}")
    #print(f"Best Params : {study.best_params}")

if __name__ == "__main__":
    main()