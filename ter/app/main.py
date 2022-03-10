from Ia import Ia
from Helper import Helper
import cv2

# Supprime les anciens résultats du dossier "resultats"
Helper.auto_remove_results()

# Lance l'intelligence artificielle
ia = Ia("./datas")
ia.extract_roi()
ia.make_clustering()

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
