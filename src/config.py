ensemble_config = {
    "rf": True,
    "xgb": True,
    "mnb": True,
    "mlp": True,
    "lr": True,
    "lstm": False
}

data_config = {
    "trained_models" : "../trained_models/",
    "saved_preds" : "../pred_probas/",
    "root": "../data/",
    "pre_processed" :"../data/pre-processed/",
    "audio": "../data/audio/",
    "text": "../data/text/",
    "combined": "../data/combined/",
    "IEMO": "../data/IEMOCAP_full_release/",
    "mode": 0, # 0:audio, 1: text, 2:combined
    "train" : False,
    "test": True
}