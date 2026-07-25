import argparse
import optuna
from optuna.visualization import (
    plot_slice,
    plot_contour,
    plot_param_importances,
    plot_optimization_history,
)


def analyze_study(study_name: str, storage: str, minimize: bool = False):
    study = optuna.load_study(study_name=study_name, storage=storage)

    print(f"Anzahl Trials: {len(study.trials)}")

    is_multi_objective = len(study.directions) > 1

    if is_multi_objective:
        print(f"Multi-Objective Study mit {len(study.directions)} Zielgrößen: {study.directions}")
        print(f"Anzahl Pareto-optimaler Trials: {len(study.best_trials)}\n")
        for t in study.best_trials:
            print(f"Trial {t.number}: values={t.values}, params={t.params}")
        print()
        # Für Slice/Contour/Importance muss bei Multi-Objective das Target angegeben werden
        target_idx = 0  # ggf. anpassen: welche Zielgröße dich interessiert
        target = lambda t: t.values[target_idx]
        target_name = f"Objective {target_idx}"
    else:
        print(f"Bester Wert: {study.best_value}")
        print(f"Beste Params: {study.best_params}\n")
        target = None
        target_name = "Objective Value"

    # --- Diagnose der fehlgeschlagenen Trials ---
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\nAnzahl FAILed Trials: {len(failed_trials)}")
    for t in failed_trials:
        fail_reason = t.system_attrs.get("fail_reason", "kein fail_reason vorhanden")
        print(f"\n--- Trial {t.number} (FAIL) ---")
        print(f"  params: {t.params}")
        print(f"  user_attrs: {t.user_attrs}")
        print(f"  fail_reason: {fail_reason}")
    print()
    # --------------------------------------------

    fig1 = plot_slice(study, params=["learning_rate", "momentum"], target=target, target_name=target_name)
    fig1.write_html("examples/har_classifier/logs/slice_plot.html")

    fig2 = plot_contour(study, params=["learning_rate", "momentum"], target=target, target_name=target_name)
    fig2.write_html("examples/har_classifier/logs/contour_plot.html")

    fig3 = plot_param_importances(study, target=target, target_name=target_name)
    fig3.write_html("examples/har_classifier/logs/param_importances.html")

    fig4 = plot_optimization_history(study, target=target, target_name=target_name)
    fig4.write_html("examples/har_classifier/logs/optimization_history.html")

    print("Plots gespeichert: slice_plot.html, contour_plot.html, "
          "param_importances.html, optimization_history.html")

    df = study.trials_dataframe()
    if not is_multi_objective:
        ascending = True if minimize else False
        top20 = df.sort_values("value", ascending=ascending).head(20)
        print("\nTop 20 Trials:")
        print(top20[["number", "value", "params_learning_rate", "params_momentum"]])
    else:
        print("\nAlle Trials (values-Spalten prüfen für Multi-Objective):")
        print(df.head(20))


def main():
    parser = argparse.ArgumentParser(description="Analysiere eine Optuna-Study")
    parser.add_argument("--study_name", type=str, required=True,
                        help="Name der Optuna-Study")
    parser.add_argument("--storage", type=str, required=True,
                        help="Storage-URL, z.B. sqlite:///dein_pfad.db")
    parser.add_argument("--minimize", action="store_true",
                        help="Setzen, falls die Study minimiert statt maximiert")
    args = parser.parse_args()

    analyze_study(args.study_name, args.storage, args.minimize)

if __name__ == "__main__":
    main()