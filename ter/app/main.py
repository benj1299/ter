from Ia import Ia
from Helper import Helper

# Supprime les anciens résultats du dossier "resultats"
Helper.auto_remove_results()

# Lance l'intelligence artificielle
ia = Ia("./datas")
ia.extract_roi()
ia.make_clustering()
