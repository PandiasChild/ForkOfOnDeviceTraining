from __future__ import annotations

import logging
import optuna
import subprocess
import string
from pathlib import Path
import json
import requests
#NOTE: c-sym-weights vs c-delta-weights for plotting!

# you need to uv add kaleido and uv add optuna
HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
OPTUNA_LOGS = LOGS/ "optuna_logs"


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





def objective(trial) -> int| float:
    trial_number = trial.number
    delta_reduction = trial.suggest_int("delta_reduction", 1, 4, step=1)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, step=0.005) #0.001
    momentum = trial.suggest_float("momentum", 0, 0.95, step=0.05) #0.9
    # rounding_mode = 0 # HALF_AWAY
    epochs = 50
    batch = 64 # möchte ich klein haben, weil für embedded device

    trial.set_user_attr("delta_reduction", delta_reduction)
    trial.set_user_attr("lr", learning_rate)
    trial.set_user_attr("momentum", momentum)
    #trial.set_user_attr("rounding_mode", rounding_mode)

    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("batch", batch)

    test_loss_delta = 0
    test_acc_delta = 0

    try:
        result = subprocess.run(
            [
                './build/examples/examples/har_classifier/train_c_delta',
                str(trial_number),
                str(delta_reduction),
                str(learning_rate),
                str(momentum),
                str(epochs),
                str(batch),
                #str(rounding_mode)
            ],
            check = True,
            capture_output=True,
            text=True
        )
        test_duration_delta = 0
        prefix = 'examples/har_classifier/logs/with_deltas/trial_'
        with open(prefix + trial_number + '.json', 'r') as f:
            data = json.load(f)

            for epochs in data["epochs"]:
                test_duration_delta += epochs.get("wall_s")

            test_loss_delta = data.get("final", {}).get("test_loss")
            test_acc_delta = data.get("final", {}).get("test_acc")

            trial.set_user_attr("test_loss_delta", test_loss_delta)
            trial.set_user_attr("test_acc_delta", test_acc_delta)
            trial.set_user_attr("test_duration_delta", test_duration_delta)

    except subprocess.CalledProcessError as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training DELTA fehlgeschlagen:\ntrial_number {trial_number}\nDas Skript ist mit Exit-Code {e.returncode} beendet worden: {e.stderr}, {e}\n"
            send_notification(bot_token, chat_id, message)

    except OSError as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nBetriebssystemfehler: {e}\n"
            send_notification(bot_token, chat_id, message)

    except FileNotFoundError:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training DELTA fehlgeschlagen:\ntrial_number {trial_number}\nPython oder das Skript wurde nicht gefunden: {e}\n"
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
    try:
        result = subprocess.run(
            [
                './build/examples/examples/har_classifier/train_c_sym',
                str(trial_number),
                str(learning_rate),
                str(momentum),
                str(epochs),
                str(batch),
                #str(rounding_mode)
            ],
            capture_output=True,
            text=True
        )
        test_duration_sym = 0
        prefix = 'examples/har_classifier/logs/without_deltas/trial_'
        with open(prefix + trial_number + '.json', 'r') as f:
            data_sym = json.load(f)

            for epochs in data_sym["epochs"]:
                test_duration_sym += epochs.get("wall_s")

            test_loss_sym = data_sym.get("final", {}).get("test_loss")
            test_acc_sym = data_sym.get("final", {}).get("test_acc")

            trial.set_user_attr("test_loss_sym", test_loss_sym)
            trial.set_user_attr("test_acc_sym", test_acc_sym)
            trial.set_user_attr("test_duration_sym", test_duration_sym)

    except subprocess.CalledProcessError as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nDas Skript ist mit Exit-Code {e.returncode} beendet worden: {e.stderr}, {e}\n"
            send_notification(bot_token, chat_id, message)

    except OSError as e:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nBetriebssystemfehler: {e}\n"
            send_notification(bot_token, chat_id, message)

    except FileNotFoundError:
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"Training SYM fehlgeschlagen:\ntrial_number {trial_number}\nPython oder das Skript wurde nicht gefunden: {e}\n"
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
    if (test_acc_delta == 0):
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"trial_number {trial_number} probably failed: test_acc_delta = 0\n"
            send_notification(bot_token, chat_id, message)

    if (test_loss_delta == 0):
        with open('telegram_bot.json', 'r') as f:
            telegram_bot = json.load(f)

            bot_token = telegram_bot.get("BOT_TOKEN", {})
            chat_id = telegram_bot.get("CHAT_ID", {})
            message = f"trial_number {trial_number} probably failed: test_loss_delta = 0\n"
            send_notification(bot_token, chat_id, message)
        return 0
    return test_acc_delta, test_loss_delta


def main():
    optuna_results_dir = OPTUNA_LOGS
    optuna_results_dir.mkdir(parents=True, exist_ok=True)

    # Create a file handler
    file_handler = logging.FileHandler(str(optuna_results_dir) + "optuna.log")
    file_handler.setLevel(logging.INFO)

    # Add it to Optuna's logger
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.addHandler(file_handler)

    # Optional: keep console quiet
    optuna.logging.disable_default_handler()

    study_name = "har_classifier_delta_vs_sym"
    study_db_path: Path = optuna_results_dir / f"{study_name}.db"

    study = optuna.create_study(
        study_name = study_name,
        directions=["maximize", "minimize"],
        storage = f"sqlite:///{study_db_path.resolve()}",
        load_if_exists=True)

    study.optimize(objective, n_trials=30, n_jobs = 2)
    space = intersection_search_space(study.get_trials())

    fig = plot_optimization_history(study)
    fig.write_html("optimization_history.html")
    fig.write_image("optimization_history.png")

    # Optimierungsverlauf
    plot_optimization_history(study).show()

    # Wichtigkeit der Hyperparameter
    plot_param_importances(study).show()

    # Parallel Coordinates
    plot_parallel_coordinate(study).show()

    # Slice Plot
    plot_slice(study).show()

    # Konturdiagramm
    plot_contour(study).show()

    # Empirical Distribution Function
    plot_edf(study).show()

    print(f"Intersection Search Space Of Trials : {space}")

    print(f"Best Trial Number : {study.best_trial.number}")
    print(f"Best Trial Value : {study.best_trial.value}")
    print(f"Best Trial Params : {study.best_trial.params}")
    print(f"Best Value : {study.best_value}")
    print(f"Best Params : {study.best_params}")

if __name__ == "__main__":
    main()