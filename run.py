from src import futures_curve, hmm_engine

# 1) rebuild/append CTSI for every curve
futures_curve.main()

# 2) update HMM probabilities
hmm_engine.main()
